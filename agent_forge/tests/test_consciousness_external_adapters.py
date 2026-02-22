from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness.external_adapters import ExternalBenchmarkImporter

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDCTL = REPO_ROOT / "agent_forge" / "bin" / "eidctl"


def test_external_benchmark_importer_persists_standardized_report(tmp_path: Path, monkeypatch) -> None:
    payload_path = tmp_path / "swe_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "resolved_rate": 0.66,
                    "consistency": 0.71,
                },
                "metadata": {"suite": "SWE-bench verified"},
            }
        ),
        encoding="utf-8",
    )
    bench_dir = tmp_path / "bench_reports"
    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(bench_dir))

    importer = ExternalBenchmarkImporter(tmp_path / "state")
    imported = importer.import_file(path=payload_path, suite="swe-bench", persist=True)
    report = imported.report

    assert imported.report_path is not None
    assert imported.report_path.exists()
    assert report.get("suite") == "swe-bench"
    assert (report.get("scores") or {}).get("composite") == 0.66
    capability = report.get("capability") or {}
    assert capability.get("coherence_ratio") == 0.66
    assert capability.get("boundary_stability") == 0.71
    assert (report.get("external") or {}).get("metric_count", 0) >= 2


def test_eidctl_consciousness_import_benchmark_command(tmp_path: Path) -> None:
    base = tmp_path / "state"
    payload_path = tmp_path / "agentbench_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "results": {"overall": 0.58},
                "success_rate": 0.61,
                "consistency": 0.62,
            }
        ),
        encoding="utf-8",
    )
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
        "import-benchmark",
        "--dir",
        str(base),
        "--path",
        str(payload_path),
        "--suite",
        "agentbench",
        "--no-persist",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    assert report.get("suite") == "agentbench"
    assert report.get("benchmark_id")
    assert (report.get("scores") or {}).get("composite") == 0.61
    assert report.get("report_path") is None
