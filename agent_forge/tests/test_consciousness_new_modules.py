from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.affect import AffectModule
from agent_forge.consciousness.modules.default_mode import DefaultModeModule
from agent_forge.consciousness.modules.metacognition import MetacognitionModule
from agent_forge.consciousness.modules.motor import MotorModule
from agent_forge.consciousness.modules.working_set import WorkingSetModule
from agent_forge.consciousness.state_store import ModuleStateStore
from agent_forge.consciousness.types import TickContext, merged_config
from agent_forge.core import events, workspace


class _Ledger:
    def __init__(self) -> None:
        self.heartbeats: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []

    def record_heartbeat(self, payload: dict[str, Any]) -> None:
        self.heartbeats.append(dict(payload))

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.history[-limit:]


def _ctx(
    state_dir: Path,
    store: ModuleStateStore,
    *,
    beat: int,
    recent_events: list[dict[str, Any]] | None = None,
    recent_broadcasts: list[dict[str, Any]] | None = None,
    global_winner: dict[str, Any] | None = None,
    ledger: _Ledger | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> TickContext:
    return TickContext(
        state_dir=state_dir,
        config=merged_config(dict(cfg or {})),
        recent_events=recent_events or [],
        recent_broadcasts=recent_broadcasts or [],
        rng=random.Random(17),
        beat_count=beat,
        state_store=store,
        active_perturbations=[],
        ledger=ledger or _Ledger(),
        global_winner=global_winner,
    )


def test_default_mode_emits_consolidation_plan_when_idle(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    ctx = _ctx(state_dir, store, beat=40)
    ctx.module_state("affect", defaults={"modulators": {}})["modulators"] = {"arousal": 0.2}
    ctx.module_state("intero", defaults={"drives": {}})["drives"] = {"threat": 0.1, "energy": 0.9}

    DefaultModeModule().tick(ctx)

    broadcasts = [evt for evt in ctx.emitted_events if evt.get("type") == "workspace.broadcast"]
    assert broadcasts
    payload = broadcasts[-1]["data"]["payload"]
    assert payload["kind"] == "PLAN"
    assert payload["content"]["action_kind"] == "consolidate_memory"
    metrics = [evt for evt in ctx.emitted_events if evt.get("type") == "metrics.sample"]
    assert any((evt.get("data") or {}).get("key") == "consciousness.dmn.activation" for evt in metrics)


def test_metacognition_emits_control_and_repair_on_high_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    ledger = _Ledger()
    ledger.history = [
        {"state_hash": "prev"},
        {"state_hash": "curr"},
    ]
    events.append(
        state_dir,
        "policy.efference",
        {
            "action_id": "a1",
            "predicted_observation": {
                "expected_kind": "REPORT",
                "expected_source": "report",
            },
        },
    )
    events.append(
        state_dir,
        "consciousness.module_error",
        {"module": "motor", "error": "boom"},
    )
    recent_events = events.iter_events(state_dir, limit=50)
    ctx = _ctx(
        state_dir,
        store,
        beat=60,
        recent_events=recent_events,
        ledger=ledger,
        global_winner={"content": {"candidate_id": "winner-1"}},
    )

    MetacognitionModule().tick(ctx)

    broadcasts = [evt["data"]["payload"] for evt in ctx.emitted_events if evt.get("type") == "workspace.broadcast"]
    kinds = [payload.get("kind") for payload in broadcasts]
    assert "METACONTROL" in kinds
    assert "PLAN" in kinds
    metacontrol = next(payload for payload in broadcasts if payload.get("kind") == "METACONTROL")
    assert metacontrol["content"]["recommendation"] == "trigger_self_repair"


def test_motor_executes_plan_and_records_heartbeat(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    ledger = _Ledger()
    monkeypatch.setattr("agent_forge.consciousness.modules.motor._gk", None)

    plan_evt = workspace.broadcast(
        state_dir,
        source="default_mode",
        payload={
            "kind": "PLAN",
            "source_module": "default_mode",
            "content": {
                "action_id": "plan-1",
                "action_kind": "consolidate_memory",
            },
        },
    )
    ctx = _ctx(
        state_dir,
        store,
        beat=80,
        recent_events=events.iter_events(state_dir, limit=50),
        recent_broadcasts=[plan_evt],
        ledger=ledger,
    )

    MotorModule().tick(ctx)

    emitted = [evt for evt in ctx.emitted_events if evt.get("type") == "motor.execution"]
    assert emitted
    motor_event = emitted[-1]["data"]
    assert motor_event["action_kind"] == "consolidate_memory_initiated"
    broadcasts = [evt["data"]["payload"] for evt in ctx.emitted_events if evt.get("type") == "workspace.broadcast"]
    assert any(payload.get("kind") == "MOTOR_INTENT" for payload in broadcasts)
    assert ledger.heartbeats
    assert ledger.heartbeats[-1]["action_kind"] == "consolidate_memory_initiated"


def test_affect_emits_extended_personality_modulators(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    ctx = _ctx(state_dir, store, beat=12)
    ctx.module_state("intero", defaults={"drives": {}})["drives"] = {
        "threat": 0.15,
        "curiosity": 0.85,
        "energy": 0.9,
        "coherence_hunger": 0.65,
        "novelty_hunger": 0.75,
    }
    ctx.module_state("phenomenology_probe", defaults={}).update({"unity_index": 0.8, "ownership_index": 0.7})
    events.append(state_dir, "metrics.sample", {"key": "consciousness.metacog.conflict_intensity", "value": 0.3})
    events.append(state_dir, "metrics.sample", {"key": "consciousness.metacog.stability", "value": 0.6})
    ctx.recent_events = events.iter_events(state_dir, limit=50)

    AffectModule().tick(ctx)

    modulators = ctx.module_state("affect", defaults={"modulators": {}})["modulators"]
    assert modulators["pride"] > 0.55
    assert modulators["ambition"] > 0.6
    assert "satisfaction" in modulators
    assert "effort" in modulators


def test_working_set_injects_global_winner_with_ignition_flag(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    ctx = _ctx(
        state_dir,
        store,
        beat=14,
        global_winner={
            "kind": "GW_WINNER",
            "source_module": "workspace_competition",
            "content": {"candidate_id": "winner-42"},
            "confidence": 0.91,
            "links": {"corr_id": "corr-42", "parent_id": "parent-42"},
            "ts": "2026-03-20T00:00:00Z",
        },
    )

    WorkingSetModule().tick(ctx)

    active_items = ctx.module_state("working_set", defaults={"active_items": []})["active_items"]
    assert any(item["item_id"] == "winner-42" and item.get("ignited") is True for item in active_items)


def test_kernel_narrative_anchor_is_configurable_and_emits_event(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    anchored: dict[str, Any] = {}

    class _Engine:
        def anchor_cycle(self, state_dir: str) -> str:
            anchored["state_dir"] = state_dir
            return "anchor ok"

        def shutdown(self) -> None:
            anchored["shutdown"] = True

    monkeypatch.setattr(ConsciousnessKernel, "_build_narrative_engine", lambda self: _Engine())
    kernel = ConsciousnessKernel(
        base,
        config={
            "kernel_narrative_anchor_enabled": True,
            "kernel_narrative_anchor_interval_beats": 1,
        },
        modules=[],
        seed=3,
    )

    result = kernel.tick()

    assert result.errors == []
    assert anchored["state_dir"] == str(base)
    emitted = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "consciousness.narrative_anchor" for evt in emitted)
    assert anchored["shutdown"] is True
