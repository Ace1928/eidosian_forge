from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness import ConsciousnessKernel, ConsciousnessTrialRunner
from agent_forge.consciousness.perturb import make_noise
from agent_forge.core import events

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def test_trial_runner_emits_perturb_events_and_rci_metric(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(base, "daemon.beat", {"rss_bytes": 123})

    runner = ConsciousnessTrialRunner(base)
    kernel = ConsciousnessKernel(base)
    result = runner.run_trial(
        kernel=kernel,
        perturbation=make_noise("attention", 0.3, 1.0),
        ticks=2,
        persist=False,
    )

    assert result.report.get("report_id")
    assert result.report.get("delta")

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "perturb.inject" for evt in all_events)
    assert any(evt.get("type") == "perturb.response" for evt in all_events)

    rci_samples = [
        evt for evt in all_events
        if evt.get("type") == "metrics.sample" and (evt.get("data") or {}).get("key") == "consciousness.rci"
    ]
    assert rci_samples


def test_trial_runner_persists_and_status_exposes_latest(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    reports = tmp_path / "reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_TRIAL_DIR", str(reports))

    runner = ConsciousnessTrialRunner(base)
    kernel = ConsciousnessKernel(base)
    trial = runner.run_trial(
        kernel=kernel,
        perturbation=make_noise("attention", 0.2, 1.0),
        ticks=1,
        persist=True,
    )

    assert trial.report_path is not None
    assert trial.report_path.exists()

    latest = runner.latest_trial()
    assert latest is not None
    assert latest.get("report_id") == trial.report_id

    status = runner.status()
    assert status.get("latest_trial") is not None
    assert "watchdog" in status
    assert "payload_safety" in status


def test_eidctl_consciousness_commands(tmp_path: Path) -> None:
    base = tmp_path / "state"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:{REPO_ROOT / 'agent_forge' / 'src'}"
    )

    status_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "status",
        "--dir",
        str(base),
        "--json",
    ]
    status_res = subprocess.run(status_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert status_res.returncode == 0, status_res.stderr
    status_payload = json.loads(status_res.stdout)
    assert "workspace" in status_payload

    trial_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "trial",
        "--dir",
        str(base),
        "--kind",
        "noise",
        "--target",
        "attention",
        "--ticks",
        "1",
        "--no-persist",
        "--json",
    ]
    trial_res = subprocess.run(trial_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert trial_res.returncode == 0, trial_res.stderr
    trial_payload = json.loads(trial_res.stdout)
    assert "report_id" in trial_payload
