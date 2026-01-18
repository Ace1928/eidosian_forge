from pathlib import Path
from core import events as E


def test_append_and_iter(tmp_path: Path):
    base = tmp_path / "state"
    E.append(base, "tick", {"n": 1})
    items = E.iter_events(base, limit=None)
    assert len(items) == 1
    assert items[0]["type"] == "tick"
    assert items[0]["data"]["n"] == 1


def test_rotation_and_bad_lines(tmp_path: Path):
    base = tmp_path / "state"
    for i in range(5):
        E.append(base, f"e{i}", max_bytes=100)
    assert E.files_count(base) >= 1
    # force rotation with tiny max_bytes
    for i in range(5, 10):
        E.append(base, f"e{i}", max_bytes=10)
    assert E.files_count(base) > 1
    # corrupt first file
    events_dir = base / "events"
    first = sorted(events_dir.rglob("bus-*.jsonl"))[0]
    with first.open("a", encoding="utf-8") as f:
        f.write("BAD LINE\n")
    items = E.iter_events(base, limit=None)
    # ensure 10 valid events counted despite bad line
    assert len([i for i in items if i["type"].startswith("e")]) == 10
