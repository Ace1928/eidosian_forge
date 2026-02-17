from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness import ConsciousnessStressBenchmark

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def test_stress_benchmark_persists_and_latest(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    stress_dir = tmp_path / "stress_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_STRESS_BENCHMARK_DIR", str(stress_dir))

    suite = ConsciousnessStressBenchmark(base)
    result = suite.run(
        ticks=2,
        event_fanout=4,
        broadcast_fanout=2,
        payload_chars=4096,
        max_payload_bytes=1024,
        persist=True,
    )

    assert result.report_path is not None
    assert result.report_path.exists()
    pressure = result.report.get("pressure") or {}
    assert int(pressure.get("payload_truncations_observed") or 0) > 0
    assert "gates" in result.report

    latest = suite.latest_stress_benchmark()
    assert latest is not None
    assert latest.get("benchmark_id") == result.benchmark_id


def test_eidctl_consciousness_stress_benchmark_commands(tmp_path: Path) -> None:
    base = tmp_path / "state"
    stress_dir = tmp_path / "stress_reports"
    env = dict(os.environ)
    env["EIDOS_CONSCIOUSNESS_STRESS_BENCHMARK_DIR"] = str(stress_dir)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:"
        f"{REPO_ROOT / 'agent_forge' / 'src'}:"
        f"{REPO_ROOT / 'crawl_forge' / 'src'}:"
        f"{REPO_ROOT / 'eidos_mcp' / 'src'}"
    )

    run_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "stress-benchmark",
        "--dir",
        str(base),
        "--ticks",
        "2",
        "--event-fanout",
        "4",
        "--broadcast-fanout",
        "2",
        "--payload-chars",
        "4096",
        "--max-payload-bytes",
        "1024",
        "--json",
    ]
    run_res = subprocess.run(run_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert run_res.returncode == 0, run_res.stderr
    run_payload = json.loads(run_res.stdout)
    assert "benchmark_id" in run_payload
    assert (run_payload.get("pressure") or {}).get("payload_truncations_observed") is not None

    latest_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "latest-stress-benchmark",
        "--dir",
        str(base),
        "--json",
    ]
    latest_res = subprocess.run(latest_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert latest_res.returncode == 0, latest_res.stderr
    latest_payload = json.loads(latest_res.stdout)
    assert latest_payload.get("benchmark_id") == run_payload.get("benchmark_id")
