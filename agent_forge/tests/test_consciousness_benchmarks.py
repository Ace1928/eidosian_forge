from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness import ConsciousnessBenchmarkSuite, ConsciousnessKernel
from agent_forge.core import events

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def test_benchmark_suite_persists_and_exposes_latest(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    bench_dir = tmp_path / "bench_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_dir))
    events.append(base, "daemon.beat", {"rss_bytes": 111})

    suite = ConsciousnessBenchmarkSuite(base)
    kernel = ConsciousnessKernel(base)
    result = suite.run(kernel=kernel, ticks=3, persist=True)

    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report["scores"]["composite"] is not None
    assert "gates" in result.report

    latest = suite.latest_benchmark()
    assert latest is not None
    assert latest.get("benchmark_id") == result.benchmark_id


def test_benchmark_supports_baseline_delta(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    bench_dir = tmp_path / "bench_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_dir))
    events.append(base, "daemon.beat", {"rss_bytes": 222})

    suite = ConsciousnessBenchmarkSuite(base)
    first = suite.run(kernel=ConsciousnessKernel(base), ticks=2, persist=True)
    second = suite.run(
        kernel=ConsciousnessKernel(base),
        ticks=2,
        persist=False,
        baseline_report=str(first.report_path),
        external_scores={"mmlu": 0.72},
        external_sources={"mmlu": "https://example.com/mmlu"},
    )
    scores = second.report.get("scores") or {}
    assert scores.get("baseline_composite") is not None
    assert scores.get("delta_composite") is not None
    external = second.report.get("external") or {}
    assert external.get("scores", {}).get("mmlu", {}).get("score") == 0.72


def test_eidctl_consciousness_benchmark_command(tmp_path: Path) -> None:
    base = tmp_path / "state"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:"
        f"{REPO_ROOT / 'agent_forge' / 'src'}:"
        f"{REPO_ROOT / 'crawl_forge' / 'src'}:"
        f"{REPO_ROOT / 'eidos_mcp' / 'src'}"
    )

    cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "benchmark",
        "--dir",
        str(base),
        "--ticks",
        "2",
        "--no-persist",
        "--external-score",
        "mmlu=0.7",
        "--external-source",
        "mmlu=https://example.com/mmlu",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert "benchmark_id" in payload
    assert payload.get("scores", {}).get("composite") is not None
