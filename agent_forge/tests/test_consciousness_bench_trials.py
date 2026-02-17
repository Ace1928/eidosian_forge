from __future__ import annotations

import json
from pathlib import Path

from agent_forge.consciousness.bench.reporting import spec_hash
from agent_forge.consciousness.bench.trials import ConsciousnessBenchRunner, TrialSpec
from agent_forge.core import events


def test_bench_runner_persists_required_artifacts(tmp_path: Path) -> None:
    base = tmp_path / "state"
    runner = ConsciousnessBenchRunner(base)
    spec = TrialSpec(
        name="prh3-smoke",
        warmup_beats=1,
        baseline_seconds=0.2,
        perturb_seconds=0.2,
        recovery_seconds=0.2,
        beat_seconds=0.1,
        task="signal_pulse",
        perturbations=[
            {
                "kind": "noise",
                "target": "attention",
                "magnitude": 0.25,
                "duration_s": 0.3,
            }
        ],
    )

    result = runner.run_trial(spec, persist=True)

    assert result.trial_id
    assert result.output_dir is not None
    assert result.output_dir.exists()
    assert (result.output_dir / "spec.json").exists()
    assert (result.output_dir / "metrics.jsonl").exists()
    assert (result.output_dir / "events_window.jsonl").exists()
    assert (result.output_dir / "module_state_snapshot.json").exists()
    assert (result.output_dir / "replay_manifest.json").exists()
    assert (result.output_dir / "summary.md").exists()
    assert (result.output_dir / "report.json").exists()
    assert result.report.get("capture_method") in {"event_id_markers", "before_count_fallback"}
    assert str(result.report.get("capture_start_event_id") or "")
    assert str(result.report.get("capture_end_event_id") or "")
    provenance = result.report.get("provenance") or {}
    assert "capture_event_digest" in provenance
    assert "capture_event_id_coverage" in provenance

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "bench.trial_result" for evt in all_events)
    assert any(evt.get("type") == "bench.trial_start" for evt in all_events)
    assert any(evt.get("type") == "bench.trial_end" for evt in all_events)
    assert any(evt.get("type") == "perturb.inject" for evt in all_events)

    window_events = [
        json.loads(line)
        for line in (result.output_dir / "events_window.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert window_events
    assert (
        str(window_events[0].get("event_id") or "")
        == str(result.report.get("capture_start_event_id") or "")
    )
    assert (
        str(window_events[-1].get("event_id") or "")
        == str(result.report.get("capture_end_event_id") or "")
    )


def test_trial_spec_hash_is_stable_and_no_persist_mode(tmp_path: Path) -> None:
    base = tmp_path / "state"
    runner = ConsciousnessBenchRunner(base)
    spec = TrialSpec(
        name="hash-check",
        warmup_beats=0,
        baseline_seconds=0.1,
        perturb_seconds=0.1,
        recovery_seconds=0.1,
        beat_seconds=0.1,
        disable_modules=["report", "meta"],
    )

    normalized = spec.normalized()
    h1 = spec_hash(normalized)
    h2 = spec_hash(dict(normalized))
    assert h1 == h2

    result = runner.run_trial(spec, persist=False)
    assert result.output_dir is None
    assert result.report.get("spec_hash") == h1
    assert "phenomenology" in (result.report.get("before") or {})
    assert "phenomenology" in (result.report.get("after") or {})
