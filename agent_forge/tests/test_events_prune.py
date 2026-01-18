from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import events as E


def test_prune_old_days(tmp_path):
    base = tmp_path / "state"
    events_dir = base / "events"
    events_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    for i in range(5):
        day = (now - timedelta(days=i + 1)).strftime("%Y%m%d")
        (events_dir / day).mkdir()
    deleted = E.prune_old_days(base, keep_days=3)
    assert deleted == 2
    remaining = sorted(p.name for p in events_dir.iterdir())
    assert len(remaining) == 3

