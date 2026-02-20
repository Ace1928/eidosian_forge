from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.state_store import ModuleStateStore
from agent_forge.consciousness.tuning.overlay import (
    load_tuned_overlay,
    persist_tuned_overlay,
    resolve_config,
    sanitize_overlay,
)
from agent_forge.consciousness.types import merged_config


def test_overlay_sanitization_and_resolution() -> None:
    overlay = {
        "competition_top_k": 3,
        "module_tick_periods": {"meta": 4},
        "bad_key": 123,
    }
    cleaned, invalid = sanitize_overlay(overlay)
    assert cleaned["competition_top_k"] == 3
    assert cleaned["module_tick_periods"]["meta"] == 4
    assert "bad_key" in invalid

    cfg = resolve_config(
        merged_config({}),
        tuned_overlay=cleaned,
        runtime_overrides={"competition_top_k": 4},
    )
    assert cfg["competition_top_k"] == 4
    assert cfg["module_tick_periods"]["meta"] == 4


def test_overlay_persistence_round_trip(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = ModuleStateStore(state_dir, autosave_interval_secs=0.0)
    res = persist_tuned_overlay(
        store,
        {
            "competition_min_score": 0.33,
            "module_tick_periods": {"report": 5},
            "unknown": "ignored",
        },
        source="test",
        reason="roundtrip",
        score=0.42,
    )
    assert res["version"] == 1
    assert "unknown" in res["invalid_keys"]
    store.flush(force=True)

    loaded, invalid = load_tuned_overlay(store)
    assert invalid == []
    assert loaded["competition_min_score"] == 0.33
    assert loaded["module_tick_periods"]["report"] == 5
    assert store.get_meta("tuned_overlay_version") == 1
