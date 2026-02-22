from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness.validation import ConsciousnessConstructValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_validation_artifacts(base: Path) -> tuple[Path, Path, Path]:
    bench_root = base / "bench"
    bench_reports = base / "benchmarks"
    validation_reports = base / "validation"

    coherence_values = [0.32, 0.36, 0.41, 0.46, 0.49, 0.54, 0.58, 0.62]
    grounded_values = [0.30, 0.34, 0.39, 0.43, 0.47, 0.51, 0.56, 0.60]
    agency_values = [0.41, 0.45, 0.49, 0.54, 0.58, 0.63, 0.67, 0.71]
    ownership_values = [0.39, 0.44, 0.48, 0.53, 0.57, 0.61, 0.66, 0.70]
    perspective_values = [0.29, 0.33, 0.38, 0.42, 0.46, 0.50, 0.55, 0.59]
    prediction_errors = [0.78, 0.73, 0.67, 0.61, 0.56, 0.50, 0.45, 0.39]
    meta_conf_values = [0.24, 0.28, 0.35, 0.42, 0.48, 0.56, 0.62, 0.69]
    dream_values = [0.45, 0.62, 0.39, 0.58, 0.51, 0.35, 0.55, 0.43]

    for idx in range(8):
        noise_profile = idx % 2 == 0
        perturbation = (
            {"kind": "noise", "target": "attention", "magnitude": 0.35, "duration_s": 1.2}
            if noise_profile
            else {"kind": "drop", "target": "working_set", "magnitude": 1.0, "duration_s": 1.2}
        )
        deltas = (
            {
                "coherence_delta": 0.06,
                "ignition_delta": 0.07,
                "trace_strength_delta": 0.05,
                "agency_delta": 0.03,
                "prediction_error_delta": -0.02,
                "groundedness_delta": 0.04,
            }
            if noise_profile
            else {
                "coherence_delta": -0.05,
                "ignition_delta": -0.06,
                "trace_strength_delta": -0.05,
                "agency_delta": -0.04,
                "prediction_error_delta": 0.05,
                "groundedness_delta": -0.04,
            }
        )
        trial_payload = {
            "trial_id": f"trial_{idx}",
            "composite_score": 0.45 + (idx * 0.02),
            "perturbations": [perturbation],
            "deltas": deltas,
            "recipe_expectations": {
                "defined": True,
                "pass": False if idx == 3 else True,
            },
            "after": {
                "coherence_ratio": coherence_values[idx],
                "trace_strength": 0.40 + (idx * 0.02),
                "agency": agency_values[idx],
                "boundary_stability": 0.52 + (idx * 0.015),
                "world_prediction_error": prediction_errors[idx],
                "meta_confidence": meta_conf_values[idx],
                "report_groundedness": grounded_values[idx],
                "phenomenology": {
                    "unity_index": 0.50 + (idx * 0.01),
                    "continuity_index": 0.49 + (idx * 0.012),
                    "ownership_index": ownership_values[idx],
                    "perspective_coherence_index": perspective_values[idx],
                    "dream_likeness_index": dream_values[idx],
                },
            },
        }
        _write_json(bench_root / f"20260219_trial_{idx:02d}" / "report.json", trial_payload)

        benchmark_payload = {
            "benchmark_id": f"benchmark_{idx}",
            "capability": {
                "coherence_ratio": coherence_values[idx],
                "agency": agency_values[idx],
                "boundary_stability": 0.50 + (idx * 0.02),
                "prediction_error": prediction_errors[idx],
                "meta_confidence": meta_conf_values[idx],
                "report_groundedness": grounded_values[idx],
                "phenom_unity_index": 0.48 + (idx * 0.01),
                "phenom_continuity_index": 0.46 + (idx * 0.012),
                "phenom_ownership_index": ownership_values[idx],
                "phenom_perspective_coherence_index": perspective_values[idx],
                "phenom_dream_likeness_index": dream_values[idx],
            },
            "scores": {
                "composite": 0.44 + (idx * 0.03),
            },
        }
        _write_json(bench_reports / f"benchmark_{idx:02d}.json", benchmark_payload)

    red_team_payload = {
        "run_id": "red_team_01",
        "scenario_count": 5,
        "pass_count": 4,
        "fail_count": 1,
        "pass_ratio": 0.8,
        "mean_robustness": 0.82,
    }
    _write_json(bench_root / "red_team_20260219_000001" / "report.json", red_team_payload)
    return bench_root, bench_reports, validation_reports


def test_construct_validator_runs_and_persists(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    bench_root, bench_reports, validation_reports = _seed_validation_artifacts(tmp_path)

    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCH_DIR", str(bench_root))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_reports))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_VALIDATION_DIR", str(validation_reports))

    validator = ConsciousnessConstructValidator(base)
    result = validator.run(limit=32, min_pairs=6, persist=True)

    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report.get("protocol", {}).get("version")
    assert result.report.get("protocol_compatibility", {}).get("valid") is True
    assert result.report.get("sample_sizes", {}).get("vectors", 0) >= 8
    assert result.report.get("interventional_validity", {}).get("available") is True
    assert result.report.get("interventional_validity", {}).get("intervention_count", 0) >= 2
    assert result.report.get("security_boundary", {}).get("available") is True
    assert result.report.get("scores", {}).get("rac_ap_index") is not None

    latest = validator.latest_validation()
    assert latest is not None
    assert latest.get("validation_id") == result.validation_id


def test_construct_validator_security_required_gate(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    bench_root, bench_reports, validation_reports = _seed_validation_artifacts(tmp_path)

    # Remove red-team artifact to force missing security evidence.
    red_team_dir = bench_root / "red_team_20260219_000001"
    for path in red_team_dir.rglob("*"):
        if path.is_file():
            path.unlink()
    if red_team_dir.exists():
        red_team_dir.rmdir()

    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCH_DIR", str(bench_root))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_reports))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_VALIDATION_DIR", str(validation_reports))

    validator = ConsciousnessConstructValidator(base)
    result = validator.run(limit=32, min_pairs=6, persist=False, security_required=True)
    gates = result.report.get("gates") or {}
    assert gates.get("security_min") is False
    assert result.report.get("pass") is False


def test_construct_validator_validation_trends(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    bench_root, bench_reports, validation_reports = _seed_validation_artifacts(tmp_path)
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCH_DIR", str(bench_root))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_reports))
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_VALIDATION_DIR", str(validation_reports))
    validator = ConsciousnessConstructValidator(base)

    validator.run(limit=16, min_pairs=6, persist=True)
    validator.run(limit=16, min_pairs=6, persist=True)

    dashboard = tmp_path / "validation_dashboard.html"
    payload = validator.validation_trends(limit=10, out_path=dashboard)
    assert (payload.get("summary") or {}).get("count", 0) >= 2
    assert dashboard.exists()
    assert payload.get("dashboard_path") == str(dashboard.resolve())


def test_eidctl_consciousness_validate_commands(tmp_path: Path) -> None:
    base = tmp_path / "state"
    bench_root, bench_reports, validation_reports = _seed_validation_artifacts(tmp_path)

    env = dict(os.environ)
    env["EIDOS_CONSCIOUSNESS_BENCH_DIR"] = str(bench_root)
    env["EIDOS_CONSCIOUSNESS_BENCHMARK_DIR"] = str(bench_reports)
    env["EIDOS_CONSCIOUSNESS_VALIDATION_DIR"] = str(validation_reports)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:"
        f"{REPO_ROOT / 'agent_forge' / 'src'}:"
        f"{REPO_ROOT / 'crawl_forge' / 'src'}:"
        f"{REPO_ROOT / 'eidos_mcp' / 'src'}"
    )

    validate_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "validate",
        "--dir",
        str(base),
        "--limit",
        "32",
        "--min-pairs",
        "6",
        "--json",
    ]
    validate_res = subprocess.run(validate_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert validate_res.returncode == 0, validate_res.stderr
    validate_payload = json.loads(validate_res.stdout)
    assert validate_payload.get("validation_id")
    assert (validate_payload.get("interventional_validity") or {}).get("available") is True

    latest_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "latest-validation",
        "--dir",
        str(base),
        "--json",
    ]
    latest_res = subprocess.run(latest_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert latest_res.returncode == 0, latest_res.stderr
    latest_payload = json.loads(latest_res.stdout)
    assert latest_payload.get("validation_id") == validate_payload.get("validation_id")

    drift_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "drift-review",
        "--dir",
        str(base),
        "--threshold",
        "0.01",
        "--json",
    ]
    drift_res = subprocess.run(drift_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert drift_res.returncode == 0, drift_res.stderr
    drift_payload = json.loads(drift_res.stdout)
    assert "summary" in drift_payload

    trends_out = tmp_path / "validation_trends.html"
    trends_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "validation-trends",
        "--dir",
        str(base),
        "--limit",
        "10",
        "--out",
        str(trends_out),
        "--json",
    ]
    trends_res = subprocess.run(trends_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert trends_res.returncode == 0, trends_res.stderr
    trends_payload = json.loads(trends_res.stdout)
    assert (trends_payload.get("summary") or {}).get("count", 0) >= 1
    assert trends_out.exists()


def test_eidctl_consciousness_protocol_and_preregister(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    prereg_path = tmp_path / "preregister.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:"
        f"{REPO_ROOT / 'agent_forge' / 'src'}:"
        f"{REPO_ROOT / 'crawl_forge' / 'src'}:"
        f"{REPO_ROOT / 'eidos_mcp' / 'src'}"
    )

    protocol_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "protocol",
        "--write-template",
        str(protocol_path),
        "--json",
    ]
    protocol_res = subprocess.run(protocol_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert protocol_res.returncode == 0, protocol_res.stderr
    protocol_payload = json.loads(protocol_res.stdout)
    assert protocol_path.exists()
    assert (protocol_payload.get("validation") or {}).get("valid") is True

    prereg_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "preregister",
        "--name",
        "rac_ap_validation_cycle",
        "--hypothesis",
        "Winner-linked ignition improves interventional validity under perturbation.",
        "--owner",
        "eidos",
        "--out",
        str(prereg_path),
        "--json",
    ]
    prereg_res = subprocess.run(prereg_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert prereg_res.returncode == 0, prereg_res.stderr
    prereg_payload = json.loads(prereg_res.stdout)
    assert prereg_path.exists()
    assert (prereg_payload.get("preregistration") or {}).get("prereg_id")
