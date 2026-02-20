from __future__ import annotations

import random
from pathlib import Path

from agent_forge.consciousness.types import TickContext, normalize_workspace_payload
from agent_forge.core import events, workspace


def _ctx(base: Path) -> TickContext:
    return TickContext(
        state_dir=base,
        config={},
        recent_events=events.iter_events(base, limit=400),
        recent_broadcasts=workspace.iter_broadcast(base, limit=400),
        rng=random.Random(7),
        beat_count=3,
    )


def test_normalize_workspace_payload_enforces_canonical_links() -> None:
    payload = {
        "kind": "GW_WINNER",
        "source_module": "workspace_competition",
        "content": {"candidate_id": "cand-1", "score": 0.91},
        "links": {"memory_ids": ["m1", "m1", "m2"]},
    }
    out = normalize_workspace_payload(
        payload,
        fallback_kind="GW_WINNER",
        source_module="workspace_competition",
    )

    links = out["links"]
    assert set(links.keys()) >= {
        "corr_id",
        "parent_id",
        "memory_ids",
        "candidate_id",
        "winner_candidate_id",
    }
    assert links["candidate_id"] == "cand-1"
    assert links["winner_candidate_id"] == "cand-1"
    assert links["memory_ids"] == ["m1", "m2"]


def test_tickcontext_broadcast_injects_corr_id_and_links(tmp_path: Path) -> None:
    base = tmp_path / "state"
    ctx = _ctx(base)

    evt = ctx.broadcast(
        "unit",
        {
            "kind": "SELF",
            "content": {"note": "link-injection-test"},
        },
    )

    assert str(evt.get("corr_id") or "")
    data = evt.get("data") or {}
    payload = data.get("payload") or {}
    links = payload.get("links") or {}
    assert links.get("corr_id") == evt.get("corr_id")
    assert "candidate_id" in links
    assert "winner_candidate_id" in links


def test_event_index_maps_corr_parent_candidate_and_winner(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "attn.candidate",
        {
            "candidate_id": "cand-A",
            "source_module": "sense",
            "source_event_type": "sense.percept",
            "kind": "PERCEPT",
            "score": 0.8,
            "links": {
                "corr_id": "corr-A",
                "parent_id": "root-A",
                "memory_ids": [],
                "candidate_id": "cand-A",
                "winner_candidate_id": "",
            },
        },
        corr_id="corr-A",
        parent_id="root-A",
    )
    workspace.broadcast(
        base,
        "workspace_competition",
        {
            "kind": "GW_WINNER",
            "source_module": "workspace_competition",
            "content": {
                "candidate_id": "cand-A",
                "winner_candidate_id": "cand-A",
                "source_module": "sense",
            },
            "links": {
                "corr_id": "corr-A",
                "parent_id": "root-A",
                "memory_ids": [],
                "candidate_id": "cand-A",
                "winner_candidate_id": "cand-A",
            },
        },
        corr_id="corr-A",
        parent_id="root-A",
    )
    events.append(
        base,
        "report.self_report",
        {"report_id": "r1", "groundedness": 0.7},
        corr_id="corr-A",
        parent_id="root-A",
    )

    ctx = _ctx(base)
    idx = ctx.index

    assert len(idx.by_type.get("attn.candidate") or []) == 1
    assert len(idx.by_type.get("workspace.broadcast") or []) == 1
    assert idx.latest_by_type.get("report.self_report") is not None
    assert len(ctx.events_by_corr_id("corr-A")) == 3
    assert len(ctx.children("root-A")) == 3
    assert ctx.candidate("cand-A") is not None
    assert len(ctx.candidate_references("cand-A")) >= 2
    winner_evt = ctx.winner_for_candidate("cand-A")
    assert winner_evt is not None
    payload = (winner_evt.get("data") or {}).get("payload") or {}
    assert payload.get("kind") == "GW_WINNER"
