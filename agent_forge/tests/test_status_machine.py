import json
import subprocess
import sqlite3
from pathlib import Path
from core import state as S
from core import scheduler as SCH


def test_status_transitions(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    SCH.STATE_DIR = str(base)
    g = S.add_goal(base, "G", "d")
    p = S.add_plan(base, g.id, "htn", {"template": "hygiene"})
    S.add_step(base, p.id, 0, "echo", json.dumps(["bash", "-lc", "exit 0"]), 1.0, "todo")
    step = S.list_steps(base)[0]

    res = SCH.act({}, step)
    step_running = S.list_steps(base)[0]
    assert step_running.status == "running"

    SCH.verify({}, step_running, res)
    step_done = S.list_steps(base)[0]
    assert step_done.status in ("ok", "fail")

    # simulate stale running
    db = base / "e3.sqlite"
    runs = S.list_runs(base, step.id)
    last_id = runs[-1].id
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE steps SET status=? WHERE id=?", ("running", step.id))
        conn.execute(
            "UPDATE runs SET started_at=?, ended_at=? WHERE id=?",
            ("2000-01-01T00:00:00Z", "2000-01-01T00:00:01Z", last_id),
        )
        conn.commit()
    finally:
        conn.close()

    subprocess.run(["python", "bin/eidosd", "--once", "--dir", str(base)], check=True)
    step_fail = S.list_steps(base)[0]
    assert step_fail.status == "fail"
