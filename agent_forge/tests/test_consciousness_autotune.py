from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from agent_forge.consciousness.modules.autotune import AutotuneModule
from agent_forge.consciousness.state_store import ModuleStateStore
from agent_forge.consciousness.types import TickContext, merged_config
from agent_forge.core import events


def _ctx(state_dir: Path, store: ModuleStateStore, beat: int, cfg: Mapping[str, Any] | None = None) -> TickContext:
    merged = {
        "autotune_enabled": True,
        "autotune_interval_beats": 1,
        "autotune_min_improvement": 0.01,
        "autotune_persist_trials": False,
        "autotune_run_red_team": False,
    }
    if cfg:
        merged.update(dict(cfg))
    return TickContext(
        state_dir=state_dir,
        config=merged_config(merged),
        recent_events=events.iter_events(state_dir, limit=400),
        recent_broadcasts=[],
        rng=random.Random(7),
        beat_count=beat,
        state_store=store,
        active_perturbations=[],
    )


def test_autotune_commits_on_improvement(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = AutotuneModule()
    reports = [
        {
            "trial_id": "baseline",
            "composite_score": 0.20,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
        {
            "trial_id": "candidate",
            "composite_score": 0.31,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 3,
            "ignitions_without_trace": 0,
        },
    ]

    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: reports.pop(0),
    )
    module.tick(_ctx(state_dir, store, beat=200))

    assert int(store.get_meta("tuned_overlay_version", 0)) == 1
    all_events = events.iter_events(state_dir, limit=None)
    assert any(evt.get("type") == "tune.commit" for evt in all_events)


def test_autotune_rolls_back_on_guardrail_failure(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = AutotuneModule()

    # First pass sets a baseline.
    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: {
            "trial_id": "baseline",
            "composite_score": 0.25,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
    )
    module.tick(_ctx(state_dir, store, beat=100))

    # Second pass proposes a candidate but fails guardrails.
    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: {
            "trial_id": "candidate",
            "composite_score": 0.40,
            "module_error_count": 2,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
    )
    module.tick(_ctx(state_dir, store, beat=200))

    assert int(store.get_meta("tuned_overlay_version", 0)) == 0
    all_events = events.iter_events(state_dir, limit=None)
    assert any(evt.get("type") == "tune.rollback" for evt in all_events)


def test_autotune_bayes_mode_tracks_history(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = AutotuneModule()
    reports = [
        {
            "trial_id": "baseline",
            "composite_score": 0.10,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 1,
            "ignitions_without_trace": 0,
            "deltas": {"coherence_delta": 0.01, "groundedness_delta": 0.01},
        },
        {
            "trial_id": "candidate",
            "composite_score": 0.25,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
            "deltas": {"coherence_delta": 0.02, "groundedness_delta": 0.03},
        },
    ]

    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: reports.pop(0),
    )
    module.tick(
        _ctx(
            state_dir,
            store,
            beat=120,
            cfg={"autotune_optimizer": "bayes_pareto"},
        )
    )
    state = store.namespace("autotune")
    optimizer_state = state.get("optimizer_state")
    assert isinstance(optimizer_state, dict)
    assert optimizer_state.get("optimizer_kind") == "bayes_pareto"
    history = optimizer_state.get("history")
    assert isinstance(history, list)
    assert len(history) >= 1


def test_autotune_red_team_guard_blocks_commit(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = AutotuneModule()
    reports = [
        {
            "trial_id": "baseline",
            "composite_score": 0.20,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
        {
            "trial_id": "candidate",
            "composite_score": 0.42,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
    ]
    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: reports.pop(0),
    )
    monkeypatch.setattr(
        module,
        "_run_red_team_guard",
        lambda *_args, **_kwargs: (
            False,
            ["red_team_pass_ratio"],
            {
                "enabled": True,
                "available": True,
                "run_id": "rt_fail",
                "pass_ratio": 0.50,
                "mean_robustness": 0.60,
            },
        ),
    )

    module.tick(_ctx(state_dir, store, beat=240, cfg={"autotune_run_red_team": True}))

    assert int(store.get_meta("tuned_overlay_version", 0)) == 0
    all_events = events.iter_events(state_dir, limit=None)
    rollback = [evt for evt in all_events if evt.get("type") == "tune.rollback"]
    assert rollback
    reasons = (rollback[-1].get("data") or {}).get("reasons") or []
    assert "red_team_pass_ratio" in reasons


def test_autotune_red_team_guard_allows_commit(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    module = AutotuneModule()
    reports = [
        {
            "trial_id": "baseline",
            "composite_score": 0.10,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 1,
            "ignitions_without_trace": 0,
        },
        {
            "trial_id": "candidate",
            "composite_score": 0.35,
            "module_error_count": 0,
            "degraded_mode_ratio": 0.0,
            "winner_count": 2,
            "ignitions_without_trace": 0,
        },
    ]
    monkeypatch.setattr(
        module,
        "_run_micro_trial",
        lambda *_args, **_kwargs: reports.pop(0),
    )
    monkeypatch.setattr(
        module,
        "_run_red_team_guard",
        lambda *_args, **_kwargs: (
            True,
            [],
            {
                "enabled": True,
                "available": True,
                "run_id": "rt_pass",
                "pass_ratio": 1.0,
                "mean_robustness": 1.0,
            },
        ),
    )

    module.tick(_ctx(state_dir, store, beat=260, cfg={"autotune_run_red_team": True}))

    assert int(store.get_meta("tuned_overlay_version", 0)) == 1
    all_events = events.iter_events(state_dir, limit=None)
    commits = [evt for evt in all_events if evt.get("type") == "tune.commit"]
    assert commits
    payload = commits[-1].get("data") or {}
    red_team = payload.get("red_team") or {}
    assert red_team.get("run_id") == "rt_pass"
