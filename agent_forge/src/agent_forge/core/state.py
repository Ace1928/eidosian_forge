#!/usr/bin/env python3
"""
Minimal state layer for Eidos E3.

- state/ layout assumed by bootstrap:
    state/
      events/
      vector_store/
      weights/
      adapters/
      snaps/
      meta/ (created on demand)

- JSONL journal at: state/events/journal.jsonl
- Schema/version marker: state/meta/version.json

Public API (stdlib only):
    migrate(base="state") -> int
    append_journal(base, text, *, etype="note", tags=None, extra=None) -> dict
    snapshot(base="state", *, last=5) -> dict
    save_snapshot(base="state", snap=None, name=None) -> Path
    iter_journal(base, *, etype=None, tag=None, since=None, until=None, limit=10) -> list[dict]
"""

from __future__ import annotations
import dataclasses as dc
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .contracts import Goal, Plan, Step, Run

__all__ = [
    "migrate",
    "append_journal",
    "snapshot",
    "save_snapshot",
    "iter_journal",
    "rotate_journal",
    "prune_journal",
    "load_snapshot",
    "diff_snapshots",
    # entity DAL
    "add_goal",
    "list_goals",
    "add_plan",
    "list_plans",
    "add_step",
    "list_steps",
    "list_steps_for_goal",
    "add_run",
    "list_runs",
]

SCHEMA_VERSION = 1

# ---------- internal paths ----------

def _p(base: Path) -> Dict[str, Path]:
    return {
        "base": base,
        "events": base / "events",
        "meta": base / "meta",
        "snaps": base / "snaps",
        "journal": base / "events" / "journal.jsonl",
        "version": base / "meta" / "version.json",
    }

def _ensure_dirs(base: Path) -> None:
    p = _p(base)
    p["events"].mkdir(parents=True, exist_ok=True)
    p["meta"].mkdir(parents=True, exist_ok=True)

# ---------- migrations ----------

def migrate(base: str | Path = "state") -> int:
    """Ensure directory structure and bump/create version marker if absent."""
    b = Path(base)
    _ensure_dirs(b)
    paths = _p(b)
    vp = paths["version"]
    jp = paths["journal"]
    if not jp.exists():
        jp.touch()
    if not vp.exists():
        vp.write_text(json.dumps({"schema": SCHEMA_VERSION, "created_at": _now_iso()}), encoding="utf-8")
        return SCHEMA_VERSION
    try:
        meta = json.loads(vp.read_text(encoding="utf-8"))
        return int(meta.get("schema", SCHEMA_VERSION))
    except Exception:
        # if corrupted, do not overwrite silently; surface but return current
        raise RuntimeError(f"Corrupted version file: {vp}")

# ---------- journal ----------

def append_journal(
    base: str | Path,
    text: str,
    *,
    etype: str = "note",
    tags: List[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Append a single journal event as JSONL and return the event dict."""
    b = Path(base)
    _ensure_dirs(b)
    evt = {
        "ts": _now_iso(),
        "type": str(etype),
        "text": str(text),
        "tags": list(tags or []),
        "extra": dict(extra or {}),
    }
    jp = _p(b)["journal"]
    with jp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    return evt


def iter_journal(
    base: str | Path,
    *,
    etype: str | None = None,
    tag: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = 10,
) -> List[Dict[str, Any]]:
    """Return journal events matching filters (AND)."""
    b = Path(base)
    p = _p(b)["journal"]
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = str(evt.get("ts", ""))
            if since and t < since:
                continue
            if until and t > until:
                continue
            if etype and str(evt.get("type", "")) != etype:
                continue
            if tag and tag not in (evt.get("tags") or []):
                continue
            out.append(evt)
    if limit is not None and limit >= 0:
        out = out[-limit:]
    return out

def rotate_journal(
    base: str | Path,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    force: bool = False,
) -> Path | None:
    """Rotate events/journal.jsonl if size exceeds ``max_bytes`` (or always if ``force``)."""
    b = Path(base)
    _ensure_dirs(b)
    p = _p(b)
    jp = p["journal"]
    if not jp.exists():
        return None
    size = jp.stat().st_size
    if not force and size <= max_bytes:
        return None
    ts = _now_iso().replace(":", "").replace("+00:00", "Z").replace("+", "Z")
    rot = p["events"] / f"journal-{ts}.jsonl"
    jp.rename(rot)
    jp.touch()
    return rot


def prune_journal(base: str | Path, *, max_bytes: int = 5 * 1024 * 1024) -> Path | None:
    """Rotate journal if it exceeds ``max_bytes``; return rotated path or ``None``."""
    return rotate_journal(base, max_bytes=max_bytes)


def load_snapshot(path: str | Path) -> Dict[str, Any]:
    """Read a snapshot JSON file into a dict."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def diff_snapshots(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a minimal diff focusing on totals; positive numbers mean increases."""
    at = dict(a.get("totals", {}))
    bt = dict(b.get("totals", {}))
    keys = set(at) | set(bt)
    delta = {k: int(bt.get(k, 0)) - int(at.get(k, 0)) for k in sorted(keys)}
    return {
        "delta_totals": delta,
        "from": {"generated_at": a.get("generated_at"), "schema": a.get("schema")},
        "to": {"generated_at": b.get("generated_at"), "schema": b.get("schema")},
    }

# ---------- entity DAL ----------


def _db(base: str | Path) -> Path:
    b = Path(base)
    _ensure_dirs(b)
    db_path = b / "e3.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY, title TEXT, drive TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, goal_id TEXT, kind TEXT, meta TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS steps(id TEXT PRIMARY KEY, plan_id TEXT, idx INTEGER, name TEXT, cmd TEXT, budget_s REAL, status TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, step_id TEXT, started_at TEXT, ended_at TEXT, rc INTEGER, bytes_out INTEGER, notes TEXT)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_plan ON steps(plan_id, idx)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_step ON runs(step_id, started_at)"
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _new_id() -> str:
    return uuid.uuid4().hex


def add_goal(base: str | Path, title: str, drive: str, *, id: str | None = None, created_at: str | None = None) -> Goal:
    g = Goal(id or _new_id(), title, drive, created_at or _now_iso())
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO goals(id, title, drive, created_at) VALUES (?,?,?,?)",
            (g.id, g.title, g.drive, g.created_at),
        )
        conn.commit()
        return g
    finally:
        conn.close()


def list_goals(base: str | Path) -> List[Goal]:
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute("SELECT id, title, drive, created_at FROM goals ORDER BY created_at")
        return [Goal(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def add_plan(
    base: str | Path,
    goal_id: str,
    kind: str,
    meta: Mapping[str, Any] | None = None,
    *,
    id: str | None = None,
    created_at: str | None = None,
) -> Plan:
    p = Plan(id or _new_id(), goal_id, kind, dict(meta or {}), created_at or _now_iso())
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO plans(id, goal_id, kind, meta, created_at) VALUES (?,?,?,?,?)",
            (p.id, p.goal_id, p.kind, json.dumps(p.meta), p.created_at),
        )
        conn.commit()
        return p
    finally:
        conn.close()


def list_plans(base: str | Path, goal_id: str | None = None) -> List[Plan]:
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        if goal_id:
            cur = conn.execute(
                "SELECT id, goal_id, kind, meta, created_at FROM plans WHERE goal_id=? ORDER BY created_at",
                (goal_id,),
            )
        else:
            cur = conn.execute(
                "SELECT id, goal_id, kind, meta, created_at FROM plans ORDER BY created_at"
            )
        return [Plan(row[0], row[1], row[2], json.loads(row[3] or "{}"), row[4]) for row in cur.fetchall()]
    finally:
        conn.close()


def add_step(
    base: str | Path,
    plan_id: str,
    idx: int,
    name: str,
    cmd: str,
    budget_s: float,
    status: str,
    *,
    id: str | None = None,
    created_at: str | None = None,
) -> Step:
    s = Step(id or _new_id(), plan_id, int(idx), name, cmd, float(budget_s), status, created_at or _now_iso())
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO steps(id, plan_id, idx, name, cmd, budget_s, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (s.id, s.plan_id, s.idx, s.name, s.cmd, s.budget_s, s.status, s.created_at),
        )
        conn.commit()
        return s
    finally:
        conn.close()


def list_steps(base: str | Path, plan_id: str | None = None) -> List[Step]:
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        if plan_id:
            cur = conn.execute(
                "SELECT id, plan_id, idx, name, cmd, budget_s, status, created_at FROM steps WHERE plan_id=? ORDER BY idx",
                (plan_id,),
            )
        else:
            cur = conn.execute(
                "SELECT id, plan_id, idx, name, cmd, budget_s, status, created_at FROM steps ORDER BY plan_id, idx"
            )
        return [Step(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def list_steps_for_goal(base: str | Path, goal_id: str) -> List[Step]:
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute(
            """
            SELECT s.id, s.plan_id, s.idx, s.name, s.cmd, s.budget_s, s.status, s.created_at
            FROM steps s JOIN plans p ON s.plan_id = p.id
            WHERE p.goal_id=? ORDER BY s.idx
            """,
            (goal_id,),
        )
        return [Step(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def add_run(
    base: str | Path,
    step_id: str,
    started_at: str,
    ended_at: str | None,
    rc: int | None,
    bytes_out: int,
    notes: str,
    *,
    id: str | None = None,
) -> Run:
    r = Run(id or _new_id(), step_id, started_at, ended_at, rc, int(bytes_out), notes)
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO runs(id, step_id, started_at, ended_at, rc, bytes_out, notes) VALUES (?,?,?,?,?,?,?)",
            (r.id, r.step_id, r.started_at, r.ended_at, r.rc, r.bytes_out, r.notes),
        )
        conn.commit()
        return r
    finally:
        conn.close()


def list_runs(base: str | Path, step_id: str | None = None) -> List[Run]:
    db = _db(base)
    conn = sqlite3.connect(db)
    try:
        if step_id:
            cur = conn.execute(
                "SELECT id, step_id, started_at, ended_at, rc, bytes_out, notes FROM runs WHERE step_id=? ORDER BY started_at",
                (step_id,),
            )
        else:
            cur = conn.execute(
                "SELECT id, step_id, started_at, ended_at, rc, bytes_out, notes FROM runs ORDER BY started_at"
            )
        return [Run(*row) for row in cur.fetchall()]
    finally:
        conn.close()

# ---------- snapshot ----------

def snapshot(base: str | Path = "state", *, last: int = 5) -> Dict[str, Any]:
    """Compute a light snapshot: counts by entity family, last ``N`` journal entries, file counts."""
    b = Path(base)
    _ensure_dirs(b)
    version = migrate(b)  # idempotent

    # parse journal (if any)
    journal_path = _p(b)["journal"]
    last_events: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"goal": 0, "plan": 0, "step": 0, "run": 0, "metric": 0, "journal": 0, "note": 0}
    if journal_path.exists():
        buf: List[Dict[str, Any]] = []
        with journal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = str(evt.get("type", "note"))
                family = etype.split(".", 1)[0] if etype else "note"
                counts[family] = counts.get(family, 0) + 1
                buf.append(evt)
        last_events = buf[-last:] if last > 0 else []

    # files (quick health signal)
    from . import events as _events  # local import to avoid circular

    files = {
        "events": _file_count(_p(b)["events"]),
        "bus": _events.files_count(b),
        "vector_store": _file_count(b / "vector_store"),
        "weights": _file_count(b / "weights"),
        "adapters": _file_count(b / "adapters"),
        "snaps": _file_count(b / "snaps"),
    }

    return {
        "schema": version,
        "base": str(b.resolve()),
        "totals": counts,
        "last_events": last_events,
        "files": files,
        "generated_at": _now_iso(),
    }


def save_snapshot(
    base: str | Path = "state",
    snap: Dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """Persist a snapshot to ``state/snaps`` and return the file path."""
    b = Path(base)
    _ensure_dirs(b)
    p = _p(b)
    p["snaps"].mkdir(parents=True, exist_ok=True)
    snap = snap or snapshot(b)
    ts = snap.get("generated_at") or _now_iso()
    safe = ts.replace(":", "").replace("+00:00", "Z").replace("+", "Z")
    label = ""
    if name:
        label = "-" + "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    out = p["snaps"] / f"{safe}{label}.json"
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out

# ---------- helpers ----------

def _file_count(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob("*") if _.is_file())

def _now_iso() -> str:
    """Return UTC time in ISO-8601 with trailing 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":  # pragma: no cover
    # tiny manual smoke test
    migrate("state")
    append_journal("state", "hello world", etype="note", tags=["smoke"])
    print(json.dumps(snapshot("state"), indent=2))

