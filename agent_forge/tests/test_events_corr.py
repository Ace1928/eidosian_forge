from core import events as E


def test_corr_and_parent(tmp_path):
    base = tmp_path / "state"
    e1 = E.append(base, "root")
    assert "corr_id" in e1 and e1["parent_id"] == e1["corr_id"]

    e2 = E.append(base, "child", parent_id=e1["corr_id"])
    assert e2["parent_id"] == e1["corr_id"]
    assert e2["corr_id"] != e1["corr_id"]

    events = E.iter_events(base)
    assert events[-2]["corr_id"] == e1["corr_id"]
    assert events[-1]["parent_id"] == e1["corr_id"]
