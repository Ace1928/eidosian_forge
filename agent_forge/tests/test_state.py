from pathlib import Path
import json, time
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


def test_snapshot_save(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    path = S.save_snapshot(base)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] >= 1
    assert "totals" in data


def test_snapshot_save_named(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    path = S.save_snapshot(base, name="baseline_1")
    assert path.exists()
    assert "baseline_1" in path.name


def test_journal_tags(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    S.append_journal(base, "hello", etype="note", tags=["x", "y"])
    snap = S.snapshot(base)
    assert snap["last_events"][-1]["tags"] == ["x", "y"]


def test_iter_journal_filters(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    S.append_journal(base, "a", etype="note", tags=["x"])
    time.sleep(1)
    S.append_journal(base, "b", etype="note")
    S.append_journal(base, "c", etype="goal.created", tags=["y"])
    S.append_journal(base, "d", etype="note", tags=["y"])
    all_items = S.iter_journal(base, limit=None)
    assert len(all_items) == 4

    notes_with_y = S.iter_journal(base, etype="note", tag="y", limit=None)
    assert [e["text"] for e in notes_with_y] == ["d"]

    s_ts = all_items[1]["ts"]
    only_after = S.iter_journal(base, since=s_ts, limit=None)
    assert [e["text"] for e in only_after] == ["b", "c", "d"]

    limited = S.iter_journal(base, limit=2)
    assert len(limited) == 2


def test_migrate_idempotent(tmp_path: Path):
    base = tmp_path / "state"
    v1 = S.migrate(base)
    v2 = S.migrate(base)
    assert v1 == v2


def test_snapshot_ignores_bad_journal_lines(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    jp = (base / "events" / "journal.jsonl")
    jp.write_text('{"type":"note","text":"ok"}\nTHIS IS NOT JSON\n', encoding="utf-8")
    snap = S.snapshot(base)
    assert snap["totals"]["note"] == 1


def test_snapshot_last_param(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    for i in range(7):
        S.append_journal(base, f"n{i}")
    snap = S.snapshot(base, last=3)
    assert len(snap["last_events"]) == 3
    snap2 = S.snapshot(base, last=0)
    assert snap2["last_events"] == []


def test_rotate_and_diff(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    # small threshold ensures rotation
    jp = base / "events" / "journal.jsonl"
    jp.write_text("x" * 10, encoding="utf-8")
    rot = S.rotate_journal(base, max_bytes=5)
    assert rot and rot.exists()
    # snapshots + diff
    a = S.snapshot(base)
    S.append_journal(base, "hello", etype="note")
    b = S.snapshot(base)
    d = S.diff_snapshots(a, b)
    assert d["delta_totals"]["note"] == 1

