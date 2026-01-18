import sqlite3, subprocess, sys
from pathlib import Path

from core import db as DB
from core import events as E
from core import state as S


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
    cmd = [sys.executable, "bin/eidosd", "--state-dir", str(state_dir), "--once"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    conn = sqlite3.connect(state_dir / "e3.sqlite")
    assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] >= 1
    assert conn.execute("SELECT count(*) FROM journal").fetchone()[0] >= 1
    conn.close()

    assert E.files_count(state_dir) >= 1
    assert len(E.iter_events(state_dir, limit=None)) >= 1

    journal = S.iter_journal(state_dir, limit=None)
    assert len(journal) >= 1
