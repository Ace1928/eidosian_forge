from pathlib import Path
import sqlite3

from core import state as S


def test_crud_round_trip(tmp_path: Path):
    base = tmp_path / "state"
    g = S.add_goal(base, "T", "integrity")
    goals = S.list_goals(base)
    assert goals == [g]

    p = S.add_plan(base, g.id, "htn", {"a": 1})
    plans = S.list_plans(base, goal_id=g.id)
    assert plans == [p]

    s = S.add_step(base, p.id, 0, "s0", "echo hi", 1.0, "todo")
    steps = S.list_steps(base, plan_id=p.id)
    assert steps == [s]

    steps_for_goal = S.list_steps_for_goal(base, g.id)
    assert steps_for_goal == [s]

    r = S.add_run(base, s.id, S._now_iso(), S._now_iso(), 0, 5, "")
    runs = S.list_runs(base, step_id=s.id)
    assert runs == [r]

    conn = sqlite3.connect(base / "e3.sqlite")
    try:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM steps JOIN plans ON steps.plan_id=plans.id WHERE plans.goal_id=?",
            (g.id,),
        ).fetchone()[0]
        assert cnt == 1
    finally:
        conn.close()
