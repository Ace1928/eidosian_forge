import sqlite3

from core import db as DB


def test_metrics_index(tmp_path):
    base = tmp_path / "state"
    DB.init_db(base)
    conn = sqlite3.connect(base / "e3.sqlite")
    idxs = conn.execute("PRAGMA index_list(metrics)").fetchall()
    conn.close()
    names = [row[1] for row in idxs]
    assert "idx_metrics_key_ts" in names

