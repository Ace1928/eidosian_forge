from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.policy import PolicyModule
from agent_forge.consciousness.modules.self_model_ext import SelfModelExtModule
from agent_forge.core import events, self_model, workspace


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


def test_policy_emits_action_and_efference(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "workspace_competition", _winner_payload("cand-9"))

    kernel = ConsciousnessKernel(base, modules=[PolicyModule()], seed=7)
    result = kernel.tick()

    assert result.errors == []
    all_events = events.iter_events(base, limit=None)
    action = [evt for evt in all_events if evt.get("type") == "policy.action"]
    eff = [evt for evt in all_events if evt.get("type") == "policy.efference"]
    assert action
    assert eff
    assert eff[-1]["data"]["predicted_observation"]["expected_kind"] == "GW_WINNER"


def test_self_model_ext_emits_agency_and_boundary(tmp_path: Path) -> None:
    base = tmp_path / "state"
    action_id = "a-1"
    events.append(
        base,
        "policy.action",
        {
            "action_id": action_id,
            "action_kind": "inspect_signal",
            "selected_candidate_id": "cand-x",
        },
        corr_id=action_id,
        parent_id=action_id,
    )
    events.append(
        base,
        "policy.efference",
        {
            "action_id": action_id,
            "predicted_observation": {
                "expected_event_type": "workspace.broadcast",
                "expected_source": "workspace_competition",
                "expected_kind": "GW_WINNER",
            },
            "confidence": 0.8,
        },
        corr_id=action_id,
        parent_id=action_id,
    )
    workspace.broadcast(base, "workspace_competition", _winner_payload("cand-x"))

    kernel = ConsciousnessKernel(base, modules=[SelfModelExtModule()], seed=7)
    result = kernel.tick()

    assert result.errors == []
    all_events = events.iter_events(base, limit=None)
    agency = [evt for evt in all_events if evt.get("type") == "self.agency_estimate"]
    boundary = [evt for evt in all_events if evt.get("type") == "self.boundary_estimate"]
    assert agency
    assert boundary
    assert float(agency[-1]["data"]["agency_confidence"]) >= 0.5


def test_self_model_ext_reduces_agency_on_efference_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "state"
    action_id = "a-mismatch"
    events.append(
        base,
        "policy.efference",
        {
            "action_id": action_id,
            "predicted_observation": {
                "expected_event_type": "workspace.broadcast",
                "expected_source": "workspace_competition",
                "expected_kind": "GW_WINNER",
            },
            "confidence": 0.9,
        },
        corr_id=action_id,
        parent_id=action_id,
    )
    workspace.broadcast(
        base,
        "sense",
        {
            "kind": "PERCEPT",
            "source_module": "sense",
            "content": {"note": "mismatch"},
        },
    )

    kernel = ConsciousnessKernel(base, modules=[SelfModelExtModule()], seed=7)
    result = kernel.tick()

    assert result.errors == []
    all_events = events.iter_events(base, limit=None)
    agency_events = [evt for evt in all_events if evt.get("type") == "self.agency_estimate"]
    assert agency_events
    agency = float(agency_events[-1]["data"]["agency_confidence"])
    assert agency < 0.5


def test_self_model_snapshot_contains_consciousness_fields(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "workspace_competition", _winner_payload("cand-z"))
    events.append(base, "self.agency_estimate", {"agency_confidence": 0.75, "action_id": "ax"})
    events.append(base, "self.boundary_estimate", {"boundary_stability": 0.66})
    events.append(base, "metrics.sample", {"key": "consciousness.agency", "value": 0.75})

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    snap = self_model.snapshot(state_dir=base, memory_dir=memory_dir, last_events=20)

    consciousness = snap.get("consciousness") or {}
    agency = consciousness.get("agency") or {}
    boundary = consciousness.get("boundary") or {}

    assert "recent_winners" in consciousness
    assert agency.get("confidence") == 0.75
    assert boundary.get("stability") == 0.66
    assert isinstance(consciousness.get("recent_winners"), list)
