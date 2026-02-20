from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.meta import MetaModule
from agent_forge.consciousness.modules.report import ReportModule
from agent_forge.consciousness.modules.simulation import SimulationModule
from agent_forge.core import events, workspace


def test_simulation_module_emits_simulated_percepts_and_broadcast(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "meta.state_estimate",
        {"mode": "simulated", "confidence": 0.8},
        corr_id="c-sim",
        parent_id="p-sim",
    )
    events.append(
        base,
        "world.belief_state",
        {
            "feature_count": 5,
            "rollout_preview": [
                {"step": 1, "predicted_event_type": "sense.percept", "confidence": 0.7, "belief_top_features": []},
                {"step": 2, "predicted_event_type": "policy.action", "confidence": 0.6, "belief_top_features": []},
            ],
        },
        corr_id="c-sim",
        parent_id="p-sim",
    )

    kernel = ConsciousnessKernel(base, modules=[SimulationModule()], seed=4)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    sim_percepts = [evt for evt in all_events if evt.get("type") == "sense.simulated_percept"]
    assert sim_percepts
    sim_broadcasts = [
        evt
        for evt in all_events
        if evt.get("type") == "workspace.broadcast" and (((evt.get("data") or {}).get("source") or "") == "simulation")
    ]
    assert sim_broadcasts


def test_meta_prefers_simulated_mode_when_simulated_fraction_is_high(tmp_path: Path) -> None:
    base = tmp_path / "state"
    for idx in range(4):
        events.append(
            base,
            "sense.simulated_percept",
            {"simulated": True, "origin": "world_model.rollout", "step": idx + 1},
        )
    events.append(base, "sense.percept", {"novelty": 0.4})
    events.append(base, "world.prediction_error", {"prediction_error": 0.2})

    # Keep workspace coherence from collapsing into purely degraded mode.
    workspace.broadcast(base, "sense", {"kind": "PERCEPT", "content": {"x": 1}})
    workspace.broadcast(base, "intero", {"kind": "DRIVE", "content": {"x": 2}})

    kernel = ConsciousnessKernel(base, modules=[MetaModule()], seed=9)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    meta_events = [evt for evt in all_events if evt.get("type") == "meta.state_estimate"]
    assert meta_events
    assert meta_events[-1]["data"]["mode"] == "simulated"


def test_report_includes_simulation_context_when_active(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(
        base,
        "workspace_competition",
        {
            "kind": "GW_WINNER",
            "content": {
                "candidate_id": "cand-sim",
                "winner_candidate_id": "cand-sim",
                "source_module": "simulation",
                "source_event_type": "sense.simulated_percept",
                "score": 0.8,
            },
            "links": {
                "corr_id": "c-simr",
                "parent_id": "p-simr",
                "memory_ids": [],
                "candidate_id": "cand-sim",
                "winner_candidate_id": "cand-sim",
            },
        },
        corr_id="c-simr",
        parent_id="p-simr",
    )
    events.append(
        base,
        "policy.action",
        {
            "action_id": "a-simr",
            "action_kind": "inspect_signal",
            "selected_candidate_id": "cand-sim",
        },
        corr_id="c-simr",
        parent_id="p-simr",
    )
    events.append(base, "self.agency_estimate", {"agency_confidence": 0.9, "action_id": "a-simr"})
    events.append(base, "self.boundary_estimate", {"boundary_stability": 0.8})
    events.append(base, "meta.state_estimate", {"mode": "simulated", "confidence": 0.85})
    events.append(base, "world.prediction_error", {"prediction_error": 0.25})
    events.append(
        base,
        "sense.simulated_percept",
        {
            "simulated": True,
            "origin": "world_model.rollout",
            "predicted_event_type": "sense.percept",
            "confidence": 0.7,
        },
        corr_id="c-simr",
        parent_id="p-simr",
    )

    kernel = ConsciousnessKernel(base, modules=[ReportModule()], seed=2)
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    reports = [evt for evt in all_events if evt.get("type") == "report.self_report"]
    assert reports
    latest = reports[-1]["data"]
    assert bool((latest.get("summary") or {}).get("simulation_active")) is True
