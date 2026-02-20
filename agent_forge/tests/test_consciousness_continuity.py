from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.attention import AttentionModule
from agent_forge.consciousness.modules.working_set import WorkingSetModule
from agent_forge.consciousness.modules.workspace_competition import (
    WorkspaceCompetitionModule,
)
from agent_forge.consciousness.types import TickContext
from agent_forge.core import events, workspace


class CounterModule:
    name = "counter"

    def tick(self, ctx: TickContext) -> None:
        state = ctx.module_state(self.name, defaults={"count": 0})
        count = int(state.get("count") or 0) + 1
        state["count"] = count
        ctx.emit_event("counter.tick", {"count": count}, tags=["test", "counter"])


def _winner_payload(candidate_id: str) -> dict[str, Any]:
    return {
        "kind": "GW_WINNER",
        "ts": "2026-01-01T00:00:00Z",
        "id": f"winner-{candidate_id}",
        "source_module": "workspace_competition",
        "confidence": 0.9,
        "salience": 0.8,
        "content": {
            "candidate_id": candidate_id,
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "score": 0.85,
        },
        "links": {"corr_id": "c-winner", "parent_id": "p-winner", "memory_ids": []},
    }


def test_kernel_multirate_scheduler_respects_module_period(tmp_path: Path) -> None:
    base = tmp_path / "state"
    kernel = ConsciousnessKernel(
        base,
        modules=[CounterModule()],
        config={"module_tick_periods": {"counter": 2}},
        seed=9,
    )
    for _ in range(4):
        kernel.tick()

    all_events = events.iter_events(base, limit=None)
    ticks = [evt for evt in all_events if evt.get("type") == "counter.tick"]
    assert len(ticks) == 2


def test_module_state_persists_across_kernel_restarts(tmp_path: Path) -> None:
    base = tmp_path / "state"
    k1 = ConsciousnessKernel(base, modules=[CounterModule()], seed=3)
    k1.tick()
    k2 = ConsciousnessKernel(base, modules=[CounterModule()], seed=3)
    k2.tick()

    ns = k2.state_store.namespace("counter")
    assert int(ns.get("count") or 0) >= 2


def test_working_set_tracks_winners_and_emits_state(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "workspace_competition", _winner_payload("cand-ws"))

    kernel = ConsciousnessKernel(base, modules=[WorkingSetModule()], seed=7)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    wm_state = [evt for evt in all_events if evt.get("type") == "wm.state"]
    assert wm_state
    assert wm_state[-1]["data"]["size"] >= 1
    wm_broadcasts = [
        evt
        for evt in all_events
        if evt.get("type") == "workspace.broadcast"
        and (((evt.get("data") or {}).get("payload") or {}).get("kind") == "WM_STATE")
    ]
    assert wm_broadcasts


def test_competition_emits_reaction_trace_and_ignite(tmp_path: Path) -> None:
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
            "links": {"corr_id": "c-rx", "parent_id": "p-rx", "memory_ids": []},
            "content": {"note": "trace me"},
        },
    )
    kernel = ConsciousnessKernel(
        base,
        modules=[WorkspaceCompetitionModule()],
        config={
            "competition_top_k": 1,
            "competition_reaction_window_secs": 60,
            "competition_reaction_min_sources": 1,
            "competition_reaction_min_count": 1,
            "competition_min_score": 0.1,
            "competition_trace_strength_threshold": 0.1,
            "competition_trace_min_eval_secs": 0.0,
        },
        seed=11,
    )

    kernel.tick()
    events.append(base, "policy.action", {"action_id": "a-rx"}, corr_id="c-rx", parent_id="p-rx")
    kernel.tick()

    all_events = events.iter_events(base, limit=None)
    traces = [evt for evt in all_events if evt.get("type") == "gw.reaction_trace"]
    assert traces
    assert traces[-1]["data"]["winner_id"] == "cand-rx"
    assert int(traces[-1]["data"]["reaction_count"]) >= 1
    assert any(evt.get("type") == "gw.ignite" for evt in all_events)


def test_attention_obeys_drop_perturbation_from_event_stream(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(base, "sense.percept", {"novelty": 0.9, "prediction_error": 0.2})
    events.append(
        base,
        "perturb.inject",
        {
            "id": "pert-attn-drop",
            "kind": "drop",
            "target": "attention",
            "magnitude": 1.0,
            "duration_s": 30.0,
            "meta": {},
        },
        tags=["consciousness", "perturb"],
    )

    kernel = ConsciousnessKernel(base, modules=[AttentionModule()], seed=42)
    kernel.tick()
    all_events = events.iter_events(base, limit=None)
    assert not any(evt.get("type") == "attn.candidate" for evt in all_events)
