from core import events as E


def test_corr_and_parent(tmp_path):
    base = tmp_path / "state"
    e1 = E.append(base, "root")
    assert "corr_id" in e1 and e1["parent_id"] == e1["corr_id"]
    assert str(e1.get("event_id") or "")
    assert isinstance(e1.get("ts_ms"), int)

    e2 = E.append(base, "child", parent_id=e1["corr_id"])
    assert e2["parent_id"] == e1["corr_id"]
    assert e2["corr_id"] != e1["corr_id"]
    assert str(e2.get("event_id") or "")
    assert isinstance(e2.get("ts_ms"), int)

    events = E.iter_events(base)
    assert events[-2]["corr_id"] == e1["corr_id"]
    assert events[-1]["parent_id"] == e1["corr_id"]
    assert events[-1]["event_id"] == e2["event_id"]
