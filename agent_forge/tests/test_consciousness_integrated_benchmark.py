from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness import IntegratedStackBenchmark
from agent_forge.core import events

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def test_integrated_benchmark_persists_and_latest(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    reports = tmp_path / "integrated_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR", str(reports))
    events.append(base, "daemon.beat", {"rss_bytes": 333})

    bench = IntegratedStackBenchmark(base)
    result = bench.run(
        rounds=1,
        bench_ticks=2,
        trial_ticks=1,
        run_mcp=False,
        run_llm=False,
        run_red_team=False,
        persist=True,
    )
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report.get("scores", {}).get("integrated") is not None

    latest = bench.latest()
    assert latest is not None
    assert latest.get("benchmark_id") == result.benchmark_id


def test_integrated_benchmark_supports_baseline_delta(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    reports = tmp_path / "integrated_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR", str(reports))
    events.append(base, "daemon.beat", {"rss_bytes": 444})

    bench = IntegratedStackBenchmark(base)
    first = bench.run(
        rounds=1,
        bench_ticks=2,
        trial_ticks=1,
        run_mcp=False,
        run_llm=False,
        run_red_team=False,
        persist=True,
    )
    second = bench.run(
        rounds=1,
        bench_ticks=2,
        trial_ticks=1,
        run_mcp=False,
        run_llm=False,
        run_red_team=False,
        persist=False,
    )
    scores = second.report.get("scores") or {}
    assert scores.get("baseline") is not None
    assert scores.get("delta") is not None
    assert first.report.get("benchmark_id") != second.report.get("benchmark_id")


def test_integrated_benchmark_with_red_team_quick_mode(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "state"
    reports = tmp_path / "integrated_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR", str(reports))
    events.append(base, "daemon.beat", {"rss_bytes": 555})

    bench = IntegratedStackBenchmark(base)
    result = bench.run(
        rounds=1,
        bench_ticks=2,
        trial_ticks=1,
        run_mcp=False,
        run_llm=False,
        run_red_team=True,
        red_team_quick=True,
        red_team_max_scenarios=1,
        persist=False,
    )
    report = result.report
    assert report.get("red_team", {}).get("scenario_count") == 1
    assert report.get("scores", {}).get("red_team_score") is not None
    assert "red_team_pass_min" in (report.get("gates") or {})


def test_eidctl_consciousness_full_benchmark_commands(tmp_path: Path) -> None:
    base = tmp_path / "state"
    reports = tmp_path / "integrated_reports"
    env = dict(os.environ)
    env["EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR"] = str(reports)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'lib'}:"
        f"{REPO_ROOT / 'agent_forge' / 'src'}:"
        f"{REPO_ROOT / 'crawl_forge' / 'src'}:"
        f"{REPO_ROOT / 'eidos_mcp' / 'src'}"
    )

    full_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "full-benchmark",
        "--dir",
        str(base),
        "--rounds",
        "1",
        "--bench-ticks",
        "2",
        "--trial-ticks",
        "1",
        "--skip-mcp",
        "--skip-llm",
        "--skip-red-team",
        "--json",
    ]
    full_res = subprocess.run(full_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert full_res.returncode == 0, full_res.stderr
    full_payload = json.loads(full_res.stdout)
    assert "benchmark_id" in full_payload
    assert full_payload.get("config", {}).get("run_red_team") is False
    assert full_payload.get("scores", {}).get("integrated") is not None

    latest_cmd = [
        sys.executable,
        str(EIDCTL),
        "consciousness",
        "latest-full-benchmark",
        "--dir",
        str(base),
        "--json",
    ]
    latest_res = subprocess.run(latest_cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert latest_res.returncode == 0, latest_res.stderr
    latest_payload = json.loads(latest_res.stdout)
    assert latest_payload.get("benchmark_id") == full_payload.get("benchmark_id")
