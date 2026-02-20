#!/usr/bin/env python3
"""eidosd — daemon shim with heartbeat loop."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path as _P

from eidosian_core import eidosian

try:
    import yaml
except Exception:
    yaml = None

# add repo root to sys.path for local imports


from agent_forge.consciousness import ConsciousnessKernel  # type: ignore
from agent_forge.core import db as DB  # type: ignore
from agent_forge.core import events as E  # type: ignore
from agent_forge.core import os_metrics as OM  # type: ignore
from agent_forge.core import scheduler as SCH  # type: ignore
from agent_forge.core import state as S  # type: ignore


@eidosian()
def run_once(
    state_dir: str,
    *,
    tick_secs: float,
    cpu: OM.CpuPercent,
    kernel: ConsciousnessKernel | None = None,
) -> None:
    """Execute one beat: collect metrics, emit event, journal, and scheduler step."""
    p = OM.process_stats()
    s = OM.system_stats()
    cpu_pct = cpu.sample()
    metrics = {
        "process.rss_bytes": p.get("rss_bytes"),
        "process.cpu_user_s": p.get("cpu_user_s"),
        "process.cpu_sys_s": p.get("cpu_sys_s"),
        "process.cpu_pct": cpu_pct,
        "system.load1": s.get("load1"),
        "system.load5": s.get("load5"),
        "system.load15": s.get("load15"),
        "system.mem_total_kb": s.get("mem_total_kb"),
        "system.mem_free_kb": s.get("mem_free_kb"),
        "system.mem_available_kb": s.get("mem_available_kb"),
        "system.cpu_pct_total": s.get("cpu_pct_total"),
        "system.swap_free_kb": s.get("swap_free_kb"),
        "process.num_threads": p.get("num_threads"),
    }
    for name, val in metrics.items():
        if val is not None:
            DB.insert_metric(state_dir, name, float(val))
    payload = {
        "tick_secs": float(tick_secs),
        "rss_bytes": p.get("rss_bytes"),
        "cpu_pct": cpu_pct,
        "load1": s.get("load1"),
        "mem_available_kb": s.get("mem_available_kb"),
    }
    DB.insert_journal(state_dir, "daemon.beat", "beat")
    E.append(state_dir, "daemon.beat", payload, tags=["daemon", "beat"])
    S.append_journal(state_dir, "daemon.beat", etype="daemon.beat")
    if kernel is not None:
        try:
            kres = kernel.tick()
            E.append(
                state_dir,
                "consciousness.beat",
                {
                    "modules": kres.modules,
                    "emitted_events": kres.emitted_events,
                    "errors": kres.errors,
                },
                tags=["consciousness", "beat"],
            )
            DB.insert_metric(state_dir, "consciousness.emitted_events", float(kres.emitted_events))
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            S.append_journal(state_dir, f"consciousness beat error: {exc}", etype="consciousness.error")

    # scheduler heartbeat
    SCH.STATE_DIR = state_dir
    SCH.sense({})
    gs = S.list_goals(state_dir)
    if not gs:
        S.add_goal(state_dir, "Hygiene: format & smoke", "integrity")
        gs = S.list_goals(state_dir)
    goal = gs[0]
    steps = S.list_steps_for_goal(state_dir, goal.id)
    if not steps:
        SCH.plan({}, goal)
    else:
        running = [s for s in steps if s.status == "running"]
        if running:
            s = running[0]
            runs = S.list_runs(state_dir, s.id)
            if runs:
                last = runs[-1]
                try:
                    start = datetime.strptime(last.started_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except Exception:
                    start = datetime.now(timezone.utc)
                now = datetime.now(timezone.utc)
                if (now - start).total_seconds() > 2 * s.budget_s:
                    db = _P(state_dir) / "e3.sqlite"
                    conn = sqlite3.connect(db)
                    try:
                        conn.execute("UPDATE steps SET status=? WHERE id=?", ("fail", s.id))
                        conn.commit()
                    finally:
                        conn.close()
            return
        todo = [s for s in steps if s.status == "todo"]
        if todo:
            s = todo[0]
            if SCH.gate({}, s):
                res = SCH.act({}, s)
                SCH.verify({}, s, res)


def _load_cfg() -> dict:
    cfg_path = _P("cfg/self.yaml")
    if yaml is None or not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


@eidosian()
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eidosd", description="Minimal Eidos daemon")
    ap.add_argument("--state-dir", dest="state_dir", default="state", help="state directory")
    ap.add_argument("--dir", dest="state_dir", help="state directory")
    ap.add_argument("--once", action="store_true", help="run one cycle then exit")
    ap.add_argument("--loop", action="store_true", help="run scheduler loop")
    ap.add_argument("--tick", type=float, help="seconds between beats")
    ap.add_argument("--max-beats", type=int, default=0, help="stop after N beats (0=inf)")
    ap.add_argument("--jitter-ms", type=int, help="jitter per beat in ms")
    ap.add_argument("--max-backoff-secs", type=float, help="maximum backoff in seconds")
    args = ap.parse_args(argv)

    cfg = _load_cfg()
    daemon_cfg = cfg.get("daemon", {}) if isinstance(cfg.get("daemon", {}), dict) else {}
    consciousness_cfg = cfg.get("consciousness", {}) if isinstance(cfg.get("consciousness", {}), dict) else {}
    tick_secs = float(args.tick if args.tick is not None else daemon_cfg.get("tick_secs", 5))
    jitter_ms = int(args.jitter_ms if args.jitter_ms is not None else daemon_cfg.get("jitter_ms", 0))
    max_backoff = float(
        args.max_backoff_secs if args.max_backoff_secs is not None else daemon_cfg.get("max_backoff_secs", 30.0)
    )
    max_beats = int(args.max_beats or daemon_cfg.get("max_beats", 0))
    ret = daemon_cfg.get("retention", {})
    metrics_max = int(ret.get("metrics_per_key_max", 10000))
    events_days = int(ret.get("events_keep_days", 7))
    journal_max = int(ret.get("journal_max_bytes", 5 * 1024 * 1024))
    maint_every = int(ret.get("maintenance_every_beats", 100))

    try:
        S.migrate(args.state_dir)
        DB.init_db(args.state_dir)
    except Exception as e:
        print(f"eidosd setup error: {e}", file=sys.stderr)
        return 1

    kernel: ConsciousnessKernel | None = None
    if bool(consciousness_cfg.get("enabled", True)):
        try:
            kernel_cfg = consciousness_cfg.get("config", {})
            if not isinstance(kernel_cfg, dict):
                kernel_cfg = {}
            kernel_seed = int(consciousness_cfg.get("seed", 1337))
            kernel = ConsciousnessKernel(args.state_dir, config=kernel_cfg, seed=kernel_seed)
        except Exception as e:
            S.append_journal(args.state_dir, f"consciousness init error: {e}", etype="consciousness.error")

    cpu = OM.CpuPercent()
    if args.once:
        try:
            run_once(args.state_dir, tick_secs=tick_secs, cpu=cpu, kernel=kernel)
            return 0
        except Exception as e:
            print(f"eidosd run error: {e}", file=sys.stderr)
            return 2
    if args.loop:
        token = SCH.StopToken()
        SCH.install_sigint(token)
        cfg_obj = SCH.BeatCfg(
            tick_secs=tick_secs,
            jitter_ms=jitter_ms,
            max_backoff_secs=max_backoff,
            max_beats=max_beats,
        )
        beats = 0

        def _beat() -> None:
            nonlocal beats
            run_once(args.state_dir, tick_secs=tick_secs, cpu=cpu, kernel=kernel)
            beats += 1
            if beats % maint_every == 0:
                try:
                    DB.prune_metrics(args.state_dir, per_key_max=metrics_max)
                    E.prune_old_days(args.state_dir, keep_days=events_days)
                    S.prune_journal(args.state_dir, max_bytes=journal_max)
                    DB.insert_journal(args.state_dir, "daemon.maintenance", "maintenance")
                    S.append_journal(args.state_dir, "daemon maintenance", etype="daemon.maintenance")
                except Exception as e:
                    S.append_journal(args.state_dir, f"maintenance error: {e}", etype="daemon.maintenance")
                beats = 0

        try:
            SCH.run_loop(cfg_obj, _beat, stop=token)
            return 0
        except Exception as e:
            print(f"eidosd loop error: {e}", file=sys.stderr)
            return 2
    ap.error("must specify --once or --loop")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
