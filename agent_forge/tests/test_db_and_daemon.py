import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from core import db as DB
from core import events as E
from core import state as S

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDOSD = REPO_ROOT / "agent_forge" / "bin" / "eidosd"
PYTHONPATH = ":".join(
    [
        str(REPO_ROOT / "lib"),
        str(REPO_ROOT / "agent_forge" / "src"),
        str(REPO_ROOT / "memory_forge" / "src"),
        str(REPO_ROOT / "knowledge_forge" / "src"),
        str(REPO_ROOT / "code_forge" / "src"),
        str(REPO_ROOT / "eidos_mcp" / "src"),
        str(REPO_ROOT / "ollama_forge" / "src"),
        str(REPO_ROOT / "web_interface_forge" / "src"),
    ]
)


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
    cmd = [sys.executable, str(EIDOSD), "--state-dir", str(state_dir), "--once"]
    res = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": PYTHONPATH}
    )
    assert res.returncode == 0, res.stderr

    conn = sqlite3.connect(state_dir / "e3.sqlite")
    assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] >= 1
    assert conn.execute("SELECT count(*) FROM journal").fetchone()[0] >= 1
    conn.close()

    assert E.files_count(state_dir) >= 1
    assert len(E.iter_events(state_dir, limit=None)) >= 1

    journal = S.iter_journal(state_dir, limit=None)
    assert len(journal) >= 1


def test_eidosd_once_reads_local_agent_status(tmp_path: Path):
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "data" / "runtime" / "local_mcp_agent"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "status.json").write_text(
        '{"status":"success","tool_calls":2,"resource_count":1}',
        encoding="utf-8",
    )
    cmd = [sys.executable, str(EIDOSD), "--state-dir", str(state_dir), "--once"]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
    )
    assert res.returncode == 0, res.stderr

    conn = sqlite3.connect(state_dir / "e3.sqlite")
    assert conn.execute("SELECT value FROM metrics WHERE key='local_agent.tool_calls'").fetchone()[0] == 2.0
    assert conn.execute("SELECT value FROM metrics WHERE key='local_agent.resource_count'").fetchone()[0] == 1.0
    conn.close()
