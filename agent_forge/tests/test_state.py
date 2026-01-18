from pathlib import Path
from core import state as S

def test_migrate_and_snapshot(tmp_path: Path):
    base = tmp_path / "state"
    v = S.migrate(base)
    assert v >= 1
    snap = S.snapshot(base)
    assert snap["schema"] == v
    assert snap["totals"]["note"] == 0
    assert snap["files"]["events"] >= 1  # at least version/journal files

def test_journal_counts(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    S.append_journal(base, "g1", etype="goal.created")
    S.append_journal(base, "p1", etype="plan.created")
    S.append_journal(base, "s1", etype="step.completed")
    S.append_journal(base, "metric logged", etype="metric.logged")
    S.append_journal(base, "note only")  # default type note

    snap = S.snapshot(base)
    assert snap["totals"]["goal"] == 1
    assert snap["totals"]["plan"] == 1
    assert snap["totals"]["step"] == 1
    assert snap["totals"]["metric"] == 1
    assert snap["totals"]["note"] >= 1  # includes our note
    assert len(snap["last_events"]) <= 5

