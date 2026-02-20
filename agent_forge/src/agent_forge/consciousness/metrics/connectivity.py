from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .ignition_trace import event_references_winner, event_source, parse_iso_utc


def _winner_payload(evt: Mapping[str, Any]) -> tuple[str, str, str, str]:
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    winner_candidate_id = str(
        links.get("winner_candidate_id") or content.get("winner_candidate_id") or content.get("candidate_id") or ""
    )
    winner_corr_id = str(links.get("corr_id") or evt.get("corr_id") or "")
    winner_parent_id = str(links.get("parent_id") or evt.get("parent_id") or "")
    winner_origin = str(content.get("source_module") or event_source(evt) or "")
    return winner_candidate_id, winner_corr_id, winner_parent_id, winner_origin


def _edge_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0
    denom = node_count * (node_count - 1)
    return float(edge_count / max(denom, 1))


def _kosaraju_scc(nodes: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {node: set() for node in nodes}
    rgraph: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        if src not in graph:
            graph[src] = set()
            rgraph[src] = set()
        if dst not in graph:
            graph[dst] = set()
            rgraph[dst] = set()
        graph[src].add(dst)
        rgraph[dst].add(src)

    seen: set[str] = set()
    order: list[str] = []

    def dfs(node: str) -> None:
        seen.add(node)
        for nxt in graph.get(node, set()):
            if nxt not in seen:
                dfs(nxt)
        order.append(node)

    for node in graph:
        if node not in seen:
            dfs(node)

    seen.clear()
    comps: list[list[str]] = []

    def rdfs(node: str, out: list[str]) -> None:
        seen.add(node)
        out.append(node)
        for nxt in rgraph.get(node, set()):
            if nxt not in seen:
                rdfs(nxt, out)

    for node in reversed(order):
        if node in seen:
            continue
        comp: list[str] = []
        rdfs(node, comp)
        comps.append(comp)
    return comps


def effective_connectivity(
    events: Sequence[Mapping[str, Any]],
    *,
    reaction_window_secs: float = 1.5,
) -> dict[str, Any]:
    rows = list(events)
    nodes: set[str] = set()
    edges_weighted: dict[tuple[str, str], int] = defaultdict(int)

    by_corr: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for evt in rows:
        src = event_source(evt)
        if src:
            nodes.add(src)
        corr_id = str(evt.get("corr_id") or "")
        if corr_id:
            by_corr[corr_id].append(evt)

    for corr_id, corr_events in by_corr.items():
        if not corr_id:
            continue
        corr_events.sort(key=lambda evt: str(evt.get("ts") or ""))
        for prev, curr in zip(corr_events, corr_events[1:]):
            a = event_source(prev)
            b = event_source(curr)
            if a and b and a != b:
                edges_weighted[(a, b)] += 1

    winners: list[Mapping[str, Any]] = []
    for evt in rows:
        if str(evt.get("type") or "") != "workspace.broadcast":
            continue
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
        if str(payload.get("kind") or "") == "GW_WINNER":
            winners.append(evt)

    for winner in winners:
        winner_candidate_id, winner_corr_id, winner_parent_id, winner_origin = _winner_payload(winner)
        if not winner_candidate_id:
            continue
        winner_time = parse_iso_utc(winner.get("ts"))
        for evt in rows:
            if evt is winner:
                continue
            evt_time = parse_iso_utc(evt.get("ts"))
            if winner_time is not None and evt_time is not None:
                dt = (evt_time - winner_time).total_seconds()
                if dt < 0.0 or dt > max(0.0, float(reaction_window_secs)):
                    continue
            if not event_references_winner(
                evt,
                winner_candidate_id=winner_candidate_id,
                winner_corr_id=winner_corr_id,
                winner_parent_id=winner_parent_id,
            ):
                continue
            target = event_source(evt)
            if winner_origin and target and winner_origin != target:
                edges_weighted[(winner_origin, target)] += 1
            if target and target != "workspace_competition":
                edges_weighted[("workspace_competition", target)] += 1

    edge_set = set(edges_weighted.keys())
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)
    for src, dst in edge_set:
        out_degree[src] += 1
        in_degree[dst] += 1
        nodes.add(src)
        nodes.add(dst)

    node_count = len(nodes)
    edge_count = len(edge_set)
    reciprocity_pairs = 0
    for src, dst in edge_set:
        if src < dst and (dst, src) in edge_set:
            reciprocity_pairs += 1
    reciprocal_denom = max(edge_count, 1)
    reciprocity = float((2 * reciprocity_pairs) / reciprocal_denom)

    scc = _kosaraju_scc(nodes, edge_set)
    largest_scc = max((len(comp) for comp in scc), default=0)
    out_vals = list(out_degree.values())
    in_vals = list(in_degree.values())
    mean_out = (sum(out_vals) / len(out_vals)) if out_vals else 0.0
    mean_in = (sum(in_vals) / len(in_vals)) if in_vals else 0.0

    workspace_out = out_degree.get("workspace_competition", 0)
    workspace_centrality = float(workspace_out / max(node_count - 1, 1)) if node_count > 1 else 0.0

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(_edge_density(node_count, edge_count), 6),
        "reciprocity": round(reciprocity, 6),
        "mean_out_degree": round(float(mean_out), 6),
        "mean_in_degree": round(float(mean_in), 6),
        "max_out_degree": int(max(out_vals) if out_vals else 0),
        "max_in_degree": int(max(in_vals) if in_vals else 0),
        "scc_count": int(len(scc)),
        "largest_scc": int(largest_scc),
        "workspace_centrality": round(workspace_centrality, 6),
        "edges": sorted(
            [{"src": src, "dst": dst, "weight": edges_weighted[(src, dst)]} for (src, dst) in edge_set],
            key=lambda row: (row["src"], row["dst"]),
        ),
    }
