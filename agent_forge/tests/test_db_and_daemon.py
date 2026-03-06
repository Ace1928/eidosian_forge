import sqlite3
import subprocess
import sys
import json
from pathlib import Path

from core import db as DB
from core import events as E
from core import state as S

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDOSD = REPO_ROOT / "agent_forge" / "bin" / "eidosd"


def test_db_helpers(tmp_path: Path):
    base = tmp_path / "state"
    DB.insert_metric(base, "m", 1.23)
    DB.insert_journal(base, "note", "hello")
    conn = sqlite3.connect(base / "e3.sqlite")
    cur = conn.execute("SELECT key, value FROM metrics")
    assert cur.fetchone() == ("m", 1.23)
    cur2 = conn.execute("SELECT type, text FROM journal")
    assert cur2.fetchone() == ("note", "hello")
    conn.close()


def test_eidosd_once(tmp_path: Path):
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    runtime_dir = repo_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "living_pipeline_status.json").write_text(
        json.dumps({"state": "running", "phase": "word_forge", "eta_seconds": 42, "records_total": 9}),
        encoding="utf-8",
    )
    (runtime_dir / "eidos_scheduler_status.json").write_text(
        json.dumps({"state": "running", "current_task": "living_pipeline", "cycle": 3, "consecutive_failures": 0}),
        encoding="utf-8",
    )
    (runtime_dir / "forge_coordinator_status.json").write_text(
        json.dumps({"state": "running", "task": "word_forge", "active_models": [{"model": "qwen3.5:2b"}]}),
        encoding="utf-8",
    )
    (runtime_dir / "forge_runtime_trends.json").write_text(
        json.dumps(
            {
                "contract": "eidos.runtime_trends.v1",
                "entries": [
                    {"task": "word_forge", "state": "running", "active_model_count": 1, "policy": {"max_active_model_instances": 2}},
                    {"task": "sleep", "state": "idle", "active_model_count": 0, "policy": {"max_active_model_instances": 2}},
                ],
            }
        ),
        encoding="utf-8",
    )
    cmd = [sys.executable, str(EIDOSD), "--state-dir", str(state_dir), "--repo-root", str(repo_root), "--once"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr

    conn = sqlite3.connect(state_dir / "e3.sqlite")
    assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] >= 1
    assert conn.execute("SELECT count(*) FROM journal").fetchone()[0] >= 1
    conn.close()

    assert E.files_count(state_dir) >= 1
    all_events = E.iter_events(state_dir, limit=None)
    assert len(all_events) >= 1
    runtime_event = next(evt for evt in all_events if evt.get("type") == "forge.runtime")
    assert runtime_event["data"]["pipeline_phase"] == "word_forge"
    assert runtime_event["data"]["active_model_count"] == 1
    assert runtime_event["data"]["runtime_peak_active_models"] == 1
    assert runtime_event["data"]["runtime_average_active_models"] == 0.5

    journal = S.iter_journal(state_dir, limit=None)
    assert len(journal) >= 1
