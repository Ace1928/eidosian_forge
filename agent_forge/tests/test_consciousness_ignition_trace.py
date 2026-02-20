from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.metrics.ignition_trace import winner_reaction_trace
from agent_forge.consciousness.modules.workspace_competition import (
    WorkspaceCompetitionModule,
)
from agent_forge.core import events


def _ts(base: datetime, delta_ms: int) -> str:
    return (base + timedelta(milliseconds=delta_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_winner_reaction_trace_scores_linked_events() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    winner_ts = _ts(base, 0)
    rows = [
        {
            "type": "workspace.broadcast",
            "ts": winner_ts,
            "corr_id": "corr-1",
            "parent_id": "parent-1",
            "data": {
                "source": "workspace_competition",
                "payload": {
                    "kind": "GW_WINNER",
                    "content": {
                        "candidate_id": "cand-1",
                        "winner_candidate_id": "cand-1",
                    },
                    "links": {
                        "corr_id": "corr-1",
                        "parent_id": "parent-1",
                        "candidate_id": "cand-1",
                        "winner_candidate_id": "cand-1",
                        "memory_ids": [],
                    },
                },
            },
        },
        {
            "type": "policy.action",
            "ts": _ts(base, 200),
            "corr_id": "corr-1",
            "parent_id": "parent-1",
            "data": {"selected_candidate_id": "cand-1"},
        },
        {
            "type": "workspace.broadcast",
            "ts": _ts(base, 400),
            "corr_id": "corr-1",
            "parent_id": "parent-1",
            "data": {
                "source": "report",
                "payload": {
                    "kind": "REPORT",
                    "content": {
                        "summary": {"winner_candidate_id": "cand-1"},
                    },
                    "links": {
                        "corr_id": "corr-1",
                        "parent_id": "parent-1",
                        "winner_candidate_id": "cand-1",
                        "candidate_id": "",
                        "memory_ids": [],
                    },
                },
            },
        },
        {
            "type": "sense.percept",
            "ts": _ts(base, 300),
            "corr_id": "other-corr",
            "parent_id": "other-parent",
            "data": {"novelty": 0.8},
        },
    ]

    trace = winner_reaction_trace(
        rows,
        winner_candidate_id="cand-1",
        winner_corr_id="corr-1",
        winner_parent_id="parent-1",
        winner_ts=winner_ts,
        reaction_window_secs=1.5,
        target_sources=2,
        target_reactions=2,
        max_latency_ms=1500.0,
    )

    assert int(trace["reaction_count"]) >= 2
    assert int(trace["reaction_source_count"]) >= 2
    assert float(trace["trace_strength"]) > 0.4
    assert trace["time_to_first_reaction_ms"] is not None


def test_competition_emits_trace_and_ignite_on_linked_reaction(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "attn.candidate",
        {
            "candidate_id": "cand-rx",
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "kind": "PERCEPT",
            "salience": 0.9,
            "confidence": 0.9,
            "score": 0.91,
            "links": {
                "corr_id": "c-rx",
                "parent_id": "p-rx",
                "memory_ids": [],
                "candidate_id": "cand-rx",
                "winner_candidate_id": "",
            },
            "content": {"note": "trace me"},
        },
    )

    kernel = ConsciousnessKernel(
        base,
        modules=[WorkspaceCompetitionModule()],
        config={
            "competition_top_k": 1,
            "competition_min_score": 0.1,
            "competition_reaction_window_secs": 30.0,
            "competition_reaction_min_sources": 1,
            "competition_reaction_min_count": 1,
            "competition_trace_strength_threshold": 0.1,
            "competition_trace_min_eval_secs": 0.0,
        },
        seed=11,
    )

    first = kernel.tick()
    assert first.errors == []

    first_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "workspace.broadcast" for evt in first_events)
    assert not any(evt.get("type") == "gw.reaction_trace" for evt in first_events)

    events.append(
        base,
        "policy.action",
        {"action_id": "a-rx", "selected_candidate_id": "cand-rx"},
        corr_id="c-rx",
        parent_id="p-rx",
    )
    second = kernel.tick()
    assert second.errors == []

    all_events = events.iter_events(base, limit=None)
    traces = [evt for evt in all_events if evt.get("type") == "gw.reaction_trace"]
    assert traces
    trace = traces[-1]["data"]
    assert trace["winner_id"] == "cand-rx"
    assert trace["winner_corr_id"] == "c-rx"
    assert float(trace["trace_strength"]) > 0.0
    assert any(evt.get("type") == "gw.ignite" for evt in all_events)


def test_competition_trace_without_linked_reactions_does_not_ignite(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "attn.candidate",
        {
            "candidate_id": "cand-no",
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "kind": "PERCEPT",
            "salience": 0.8,
            "confidence": 0.9,
            "score": 0.87,
            "links": {
                "corr_id": "c-no",
                "parent_id": "p-no",
                "memory_ids": [],
                "candidate_id": "cand-no",
                "winner_candidate_id": "",
            },
            "content": {"note": "no ignite"},
        },
    )

    kernel = ConsciousnessKernel(
        base,
        modules=[WorkspaceCompetitionModule()],
        config={
            "competition_top_k": 1,
            "competition_min_score": 0.1,
            "competition_reaction_window_secs": 0.0,
            "competition_reaction_min_sources": 1,
            "competition_reaction_min_count": 1,
            "competition_trace_strength_threshold": 0.25,
            "competition_trace_min_eval_secs": 0.0,
        },
        seed=13,
    )
    kernel.tick()

    events.append(
        base,
        "policy.action",
        {"action_id": "a-no", "selected_candidate_id": "cand-no"},
        corr_id="other-corr",
        parent_id="other-parent",
    )
    kernel.tick()

    all_events = events.iter_events(base, limit=None)
    traces = [evt for evt in all_events if evt.get("type") == "gw.reaction_trace"]
    assert traces
    assert float(traces[-1]["data"]["trace_strength"]) <= 0.25
    assert not any(evt.get("type") == "gw.ignite" for evt in all_events)
