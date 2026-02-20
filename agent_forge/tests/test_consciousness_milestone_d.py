from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.meta import MetaModule
from agent_forge.consciousness.modules.report import ReportModule
from agent_forge.consciousness.modules.world_model import WorldModelModule
from agent_forge.core import events, workspace


def _winner_payload(candidate_id: str = "cand-1") -> dict:
    return {
        "kind": "GW_WINNER",
        "ts": "2026-01-01T00:00:00Z",
        "id": "winner-1",
        "source_module": "workspace_competition",
        "confidence": 0.9,
        "salience": 0.8,
        "content": {
            "candidate_id": candidate_id,
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "score": 0.88,
        },
        "links": {"corr_id": "c1", "parent_id": "p1", "memory_ids": []},
    }


def test_world_model_emits_prediction_error_and_pred_err_broadcast(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(base, "sense.percept", {"novelty": 0.8})
    events.append(base, "intero.drive", {"strength": 0.4})

    kernel = ConsciousnessKernel(base, modules=[WorldModelModule()], seed=7)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "world.prediction" for evt in all_events)
    assert any(evt.get("type") == "world.prediction_error" for evt in all_events)

    pred_err_broadcasts = []
    for evt in all_events:
        if evt.get("type") != "workspace.broadcast":
            continue
        payload = (evt.get("data") or {}).get("payload") or {}
        if payload.get("kind") == "PRED_ERR":
            pred_err_broadcasts.append(evt)
    assert pred_err_broadcasts


def test_world_model_emits_belief_state_and_rollout(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(base, "sense.percept", {"novelty": 0.7})
    events.append(base, "intero.drive", {"strength": 0.5})
    events.append(base, "policy.action", {"score": 0.6})

    module = WorldModelModule()
    kernel = ConsciousnessKernel(
        base,
        modules=[module],
        config={"world_prediction_window": 50, "world_rollout_default_steps": 4},
        seed=5,
    )
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    belief_events = [evt for evt in all_events if evt.get("type") == "world.belief_state"]
    assert belief_events
    latest = belief_events[-1]["data"]
    assert int(latest.get("feature_count") or 0) >= 1
    assert isinstance(latest.get("belief_top_features"), list)

    rollout = module.rollout(steps=4)
    assert len(rollout) == 4
    assert all("predicted_event_type" in row for row in rollout)


def test_meta_emits_grounded_mode_when_signals_are_stable(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "sense", {"kind": "PERCEPT", "content": {"x": 1}})
    workspace.broadcast(base, "intero", {"kind": "DRIVE", "content": {"x": 2}})
    workspace.broadcast(base, "policy", {"kind": "PLAN", "content": {"x": 3}})
    events.append(base, "world.prediction_error", {"prediction_error": 0.1})
    events.append(base, "metrics.sample", {"key": "consciousness.report.groundedness", "value": 0.9})

    kernel = ConsciousnessKernel(base, modules=[MetaModule()], seed=7)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    meta_events = [evt for evt in all_events if evt.get("type") == "meta.state_estimate"]
    assert meta_events
    assert meta_events[-1]["data"]["mode"] in {"grounded", "simulated"}

    meta_broadcasts = [
        evt
        for evt in all_events
        if evt.get("type") == "workspace.broadcast"
        and (((evt.get("data") or {}).get("payload") or {}).get("kind") == "META")
    ]
    assert meta_broadcasts


def test_report_emits_grounded_self_report(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "workspace_competition", _winner_payload("cand-z"))
    events.append(
        base,
        "policy.action",
        {
            "action_id": "a-1",
            "action_kind": "inspect_signal",
            "selected_candidate_id": "cand-z",
        },
        corr_id="a-1",
        parent_id="a-1",
    )
    events.append(base, "self.agency_estimate", {"agency_confidence": 0.9, "action_id": "a-1"})
    events.append(base, "self.boundary_estimate", {"boundary_stability": 0.82})
    events.append(base, "meta.state_estimate", {"mode": "grounded", "confidence": 0.87})
    events.append(base, "world.prediction_error", {"prediction_error": 0.2})

    kernel = ConsciousnessKernel(base, modules=[ReportModule()], seed=7)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    reports = [evt for evt in all_events if evt.get("type") == "report.self_report"]
    assert reports
    groundedness = float(reports[-1]["data"]["groundedness"])
    assert groundedness >= 0.6

    groundedness_metrics = [
        evt
        for evt in all_events
        if evt.get("type") == "metrics.sample"
        and (evt.get("data") or {}).get("key") == "consciousness.report.groundedness"
    ]
    assert groundedness_metrics
