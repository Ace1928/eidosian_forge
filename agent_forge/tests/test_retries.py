import json
from pathlib import Path
from core import state as S
from core import scheduler as SCH


def test_retries(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    SCH.STATE_DIR = str(base)
    g = S.add_goal(base, "G", "d")
    p = S.add_plan(base, g.id, "htn", {"template": "hygiene", "retries": {"0": 1}})
    # step fails first then passes
    done_path = base / "done"
    cmd = ["bash", "-lc", f"if [ -f {done_path} ]; then exit 0; else touch {done_path}; exit 1; fi"]
    S.add_step(base, p.id, 0, "flaky", json.dumps(cmd), 1.0, "todo")
    step = S.list_steps(base)[0]

    res1 = SCH.act({}, step)
    SCH.verify({}, step, res1)
    step_retry = S.list_steps(base)[0]
    assert step_retry.status == "todo"

    res2 = SCH.act({}, step_retry)
    SCH.verify({}, step_retry, res2)
    step_ok = S.list_steps(base)[0]
    assert step_ok.status == "ok"

    # step with retries=0 fails immediately
    p2 = S.add_plan(base, g.id, "htn", {"template": "hygiene", "retries": {"0": 0}})
    cmd_fail = ["bash", "-lc", "exit 1"]
    S.add_step(base, p2.id, 0, "bad", json.dumps(cmd_fail), 1.0, "todo")
    s2 = [s for s in S.list_steps(base) if s.plan_id == p2.id][0]
    r = SCH.act({}, s2)
    SCH.verify({}, s2, r)
    s2_final = [s for s in S.list_steps(base) if s.id == s2.id][0]
    assert s2_final.status == "fail"
