#!/usr/bin/env python3
"""Append-only JSONL event bus for Eidos E3.

- Events live under ``state/events/YYYYMMDD/bus-XXXX.jsonl``.
- ``events/latest`` symlink points to current file for quick tailing.
- Files rotate when they exceed ``max_bytes`` or when the date changes.
- Only stdlib is used.

Public API:
    append(base, etype, data=None, *, tags=None, max_bytes=5MB) -> dict
    iter_events(base, *, since=None, limit=100) -> list[dict]
    files_count(base) -> int
"""

from __future__ import annotations
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

__all__ = ["append", "iter_events", "files_count", "prune_old_days"]


def append(
    base: str | Path,
    etype: str,
    data: Mapping[str, Any] | None = None,
    *,
    tags: List[str] | None = None,
    corr_id: str | None = None,
    parent_id: str | None = None,
    max_bytes: int = 5 * 1024 * 1024,
) -> Dict[str, Any]:
    """Append an event to the bus and return the event dict.

    Rotation occurs automatically when ``max_bytes`` is exceeded or the
    calendar day changes. A ``latest`` symlink under ``events/`` always
    points to the active file.
    """
    b = Path(base)
    events_dir = b / "events"
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    day_dir = events_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    latest = events_dir / "latest"
    current: Path | None = latest.resolve() if latest.is_symlink() else None
    if (
        current is None
        or not current.exists()
        or current.stat().st_size >= max_bytes
        or current.parent != day_dir
    ):
        # need a new file
        idx = 0
        while True:
            cand = day_dir / f"bus-{idx:04d}.jsonl"
            if not cand.exists():
                cand.touch()
                current = cand
                break
            idx += 1
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(current)
        except OSError:
            pass

    corr_id = corr_id or uuid.uuid4().hex
    parent_id = parent_id or corr_id
    evt = {
        "ts": _now_iso(),
        "type": str(etype),
        "data": dict(data or {}),
        "tags": list(tags or []),
        "corr_id": corr_id,
        "parent_id": parent_id,
    }
    with current.open("a", encoding="utf-8") as f:  # type: ignore[arg-type]
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    return evt


def iter_events(
    base: str | Path,
    *,
    since: str | None = None,
    limit: int | None = 100,
) -> List[Dict[str, Any]]:
    """Return events matching ``since`` (ISO string) with optional limit."""
    b = Path(base)
    events_dir = b / "events"
    files = sorted(events_dir.rglob("bus-*.jsonl"))
    out: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if since and str(evt.get("ts", "")) < since:
                        continue
                    out.append(evt)
        except OSError:
            continue
    if limit is not None and limit >= 0:
        out = out[-limit:]
    return out


def files_count(base: str | Path) -> int:
    """Return number of event bus files under ``base/events``."""
    b = Path(base)
    events_dir = b / "events"
    return sum(1 for _ in events_dir.rglob("bus-*.jsonl"))


def prune_old_days(base: str | Path, *, keep_days: int = 7) -> int:
    """Delete ``events/YYYYMMDD`` directories older than ``keep_days``."""
    b = Path(base) / "events"
    today = datetime.utcnow().date()
    cutoff = today - timedelta(days=keep_days)
    deleted = 0
    if not b.exists():
        return 0
    for day_dir in b.iterdir():
        if not day_dir.is_dir():
            continue
        name = day_dir.name
        try:
            dt = datetime.strptime(name, "%Y%m%d").date()
        except ValueError:
            continue
        if dt < cutoff:
            try:
                shutil.rmtree(day_dir)
                deleted += 1
            except OSError:
                continue
    return deleted


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":  # pragma: no cover
    append("state", "smoke", {"msg": "hi"})
    print(iter_events("state"))
