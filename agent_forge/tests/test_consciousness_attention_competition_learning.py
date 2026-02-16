from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules import AttentionModule, WorkspaceCompetitionModule
from agent_forge.consciousness.state_store import ModuleStateStore
from agent_forge.core import events


def test_attention_adaptive_weights_update_from_trace_feedback(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "sense.percept",
        {"novelty": 0.95, "prediction_error": 0.82, "strength": 0.6},
    )
    kernel = ConsciousnessKernel(
        base,
        modules=[AttentionModule()],
        config={"attention_learning_rate": 0.2},
        seed=42,
    )
    kernel.tick()

    candidates = [
        evt
        for evt in events.iter_events(base, limit=None)
        if evt.get("type") == "attn.candidate"
    ]
    assert candidates
    candidate_id = str((candidates[-1].get("data") or {}).get("candidate_id") or "")
    assert candidate_id

    events.append(
        base,
        "gw.reaction_trace",
        {
            "winner_candidate_id": candidate_id,
            "winner_beat": 1,
            "reaction_count": 4,
            "trace_strength": 0.93,
        },
    )
    events.append(
        base,
        "sense.percept",
        {"novelty": 0.4, "prediction_error": 0.2, "strength": 0.2},
    )
    kernel.tick()

    store = ModuleStateStore(base, autosave_interval_secs=0.0)
    state = store.namespace("attention")
    weights = state.get("weights")
    assert isinstance(weights, dict)
    assert abs(sum(float(v) for v in weights.values()) - 1.0) < 1e-5
    assert any(
        evt.get("type") == "attn.weights_update"
        for evt in events.iter_events(base, limit=None)
    )


def test_competition_adaptive_policy_updates_from_trace_events(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "gw.reaction_trace",
        {
            "winner_candidate_id": "cand-a",
            "winner_beat": 1,
            "reaction_count": 6,
            "trace_strength": 0.9,
        },
    )
    kernel = ConsciousnessKernel(
        base,
        modules=[WorkspaceCompetitionModule()],
        config={"competition_adaptive_lr": 0.2},
        seed=11,
    )
    kernel.tick()

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "gw.policy_update" for evt in all_events)
    store = ModuleStateStore(base, autosave_interval_secs=0.0)
    ws_state = store.namespace("workspace_competition")
    adaptive = ws_state.get("adaptive")
    assert isinstance(adaptive, dict)
    assert float(adaptive.get("baseline_trace", 0.0)) > 0.45

