import sqlite3

from core import db as DB


def test_prune_metrics(tmp_path):
    base = tmp_path / "state"
    for i in range(6000):
        ts = f"{i:05d}"
        DB.insert_metric(base, "a", 1.0, ts=ts)
        DB.insert_metric(base, "b", 2.0, ts=ts)
    deleted = DB.prune_metrics(base, per_key_max=1000)
    assert deleted == 10000
    conn = sqlite3.connect(base / "e3.sqlite")
    counts = dict(conn.execute("SELECT key, count(*) FROM metrics GROUP BY key"))
    conn.close()
    assert counts == {"a": 1000, "b": 1000}

