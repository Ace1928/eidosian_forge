from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _event_source(evt: Mapping[str, Any]) -> str:
    etype = str(evt.get("type") or "")
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    if etype == "workspace.broadcast":
        src = str(data.get("source") or "")
        if src:
            return src
    if isinstance(data, Mapping):
        src = str(data.get("source_module") or data.get("source") or "")
        if src:
            return src
    return etype.split(".", 1)[0] if "." in etype else (etype or "unknown")


def _payload(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    return payload


@dataclass
class EventIndex:
    latest_by_type: dict[str, dict[str, Any]]
    latest_by_module: dict[str, dict[str, Any]]
    broadcasts_by_kind: dict[str, list[dict[str, Any]]]
    by_corr_id: dict[str, list[dict[str, Any]]]
    children_by_parent: dict[str, list[dict[str, Any]]]
    candidates_by_id: dict[str, dict[str, Any]]
    winners_by_candidate_id: dict[str, dict[str, Any]]
    references_by_candidate_id: dict[str, list[dict[str, Any]]]


def build_index(events: list[dict[str, Any]]) -> EventIndex:
    latest_by_type: dict[str, dict[str, Any]] = {}
    latest_by_module: dict[str, dict[str, Any]] = {}
    broadcasts_by_kind: dict[str, list[dict[str, Any]]] = {}
    by_corr_id: dict[str, list[dict[str, Any]]] = {}
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    winners_by_candidate_id: dict[str, dict[str, Any]] = {}
    references_by_candidate_id: dict[str, list[dict[str, Any]]] = {}

    for evt in events:
        etype = str(evt.get("type") or "")
        if etype:
            latest_by_type[etype] = evt

        module = _event_source(evt)
        if module:
            latest_by_module[module] = evt

        corr_id = str(evt.get("corr_id") or "")
        if corr_id:
            by_corr_id.setdefault(corr_id, []).append(evt)

        parent_id = str(evt.get("parent_id") or "")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(evt)

        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        refs: set[str] = set()
        candidate_id = ""
        if isinstance(data, Mapping):
            candidate_id = str(data.get("candidate_id") or "")
            winner_candidate = str(data.get("winner_candidate_id") or "")
            if candidate_id:
                refs.add(candidate_id)
            if winner_candidate:
                refs.add(winner_candidate)
        if etype == "attn.candidate":
            cid = str(data.get("candidate_id") or "") if isinstance(data, Mapping) else ""
            if cid:
                candidates_by_id[cid] = evt
                candidate_id = candidate_id or cid
                refs.add(cid)

        if etype == "workspace.broadcast":
            payload = _payload(evt)
            kind = str(payload.get("kind") or "")
            if kind:
                broadcasts_by_kind.setdefault(kind, []).append(evt)

            links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
            payload_corr = str(links.get("corr_id") or "")
            payload_parent = str(links.get("parent_id") or "")
            if payload_corr and payload_corr != corr_id:
                by_corr_id.setdefault(payload_corr, []).append(evt)
            if payload_parent and payload_parent != parent_id:
                children_by_parent.setdefault(payload_parent, []).append(evt)

            content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
            payload_candidate = str(content.get("candidate_id") or links.get("candidate_id") or "")
            winner_candidate = str(
                links.get("winner_candidate_id")
                or content.get("winner_candidate_id")
                or (payload_candidate if kind == "GW_WINNER" else "")
            )
            if payload_candidate:
                candidate_id = candidate_id or payload_candidate
                refs.add(payload_candidate)
            if winner_candidate:
                winners_by_candidate_id[winner_candidate] = evt
                refs.add(winner_candidate)

        if candidate_id:
            # Last write wins; preserves the newest candidate-relevant event.
            candidates_by_id[candidate_id] = evt
        for ref in refs:
            if ref:
                references_by_candidate_id.setdefault(ref, []).append(evt)

    return EventIndex(
        latest_by_type=latest_by_type,
        latest_by_module=latest_by_module,
        broadcasts_by_kind=broadcasts_by_kind,
        by_corr_id=by_corr_id,
        children_by_parent=children_by_parent,
        candidates_by_id=candidates_by_id,
        winners_by_candidate_id=winners_by_candidate_id,
        references_by_candidate_id=references_by_candidate_id,
    )
