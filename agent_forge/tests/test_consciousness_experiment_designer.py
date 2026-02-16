from __future__ import annotations

import random
from pathlib import Path

from agent_forge.consciousness.modules.experiment_designer import ExperimentDesignerModule
from agent_forge.consciousness.state_store import ModuleStateStore
from agent_forge.consciousness.types import TickContext, merged_config
from agent_forge.core import events


def _ctx(
    state_dir: Path,
    store: ModuleStateStore,
    *,
    beat: int,
    cfg: dict | None = None,
    active_perturbations: list[dict] | None = None,
) -> TickContext:
    config = {
        "experiment_designer_enabled": True,
        "experiment_designer_interval_beats": 1,
        "experiment_designer_min_trials": 2,
        "experiment_designer_auto_inject": False,
    }
    if cfg:
        config.update(cfg)
    return TickContext(
        state_dir=state_dir,
        config=merged_config(config),
        recent_events=events.iter_events(state_dir, limit=300),
        recent_broadcasts=[],
        rng=random.Random(9),
        beat_count=beat,
        state_store=store,
        active_perturbations=active_perturbations or [],
    )


def test_experiment_designer_proposes_recipe_from_trial_deltas(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = ExperimentDesignerModule()

    events.append(
        state_dir,
        "bench.trial_result",
        {
            "trial_id": "t1",
            "deltas": {
                "continuity_delta": -0.08,
                "coherence_delta": -0.03,
            },
        },
    )
    events.append(
        state_dir,
        "bench.trial_result",
        {
            "trial_id": "t2",
            "deltas": {
                "continuity_delta": -0.06,
                "trace_strength_delta": -0.01,
            },
        },
    )

    module.tick(_ctx(state_dir, store, beat=140))

    all_events = events.iter_events(state_dir, limit=None)
    proposed = [evt for evt in all_events if evt.get("type") == "experiment.proposed"]
    assert proposed, "experiment.proposed should be emitted"
    data = proposed[-1].get("data") or {}
    assert data.get("recipe") == "wm_lesion"
    assert "continuity" in str(data.get("hypothesis") or "").lower()


def test_experiment_designer_auto_injects_recipe_perturbations(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = ExperimentDesignerModule()

    events.append(
        state_dir,
        "bench.trial_result",
        {
            "trial_id": "t3",
            "deltas": {
                "trace_strength_delta": -0.12,
                "coherence_delta": -0.05,
            },
        },
    )

    module.tick(
        _ctx(
            state_dir,
            store,
            beat=220,
            cfg={
                "experiment_designer_min_trials": 1,
                "experiment_designer_auto_inject": True,
            },
        )
    )

    all_events = events.iter_events(state_dir, limit=None)
    injected = [evt for evt in all_events if evt.get("type") == "perturb.inject"]
    executed = [evt for evt in all_events if evt.get("type") == "experiment.executed"]
    assert injected, "auto inject should emit perturb.inject events"
    assert executed, "auto inject should emit experiment.executed"


def test_experiment_designer_skips_when_runtime_is_not_safe(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = ExperimentDesignerModule()

    events.append(
        state_dir,
        "bench.trial_result",
        {"trial_id": "t4", "deltas": {"coherence_delta": -0.04}},
    )

    module.tick(
        _ctx(
            state_dir,
            store,
            beat=300,
            cfg={"experiment_designer_min_trials": 1},
            active_perturbations=[
                {
                    "id": "p1",
                    "kind": "noise",
                    "target": "*",
                    "magnitude": 0.3,
                    "duration_s": 3.0,
                }
            ],
        )
    )

    all_events = events.iter_events(state_dir, limit=None)
    skipped = [evt for evt in all_events if evt.get("type") == "experiment.skipped"]
    proposed = [evt for evt in all_events if evt.get("type") == "experiment.proposed"]
    assert skipped
    assert not proposed
