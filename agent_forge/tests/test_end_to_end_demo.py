from pathlib import Path
from core.state import add_goal, list_steps_for_goal
import subprocess, json, time

def test_hygiene_e2e(tmp_path: Path):
    base = tmp_path / "state"; base.mkdir(parents=True, exist_ok=True)
    # bootstrap goal
    from core.state import add_goal
    g = add_goal(base, "Hygiene: format & smoke", "integrity")
    # run daemon once to plan
    subprocess.check_call(["bin/eidosd","--once","--dir",str(base)])
    steps = list_steps_for_goal(base, g.id)
    assert len(steps) >= 2
    # run again to execute first step
    subprocess.check_call(["bin/eidosd","--once","--dir",str(base)])
    steps2 = list_steps_for_goal(base, g.id)
    assert "ok" in {s.status for s in steps2} or "fail" in {s.status for s in steps2}
