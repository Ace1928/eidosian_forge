#!/usr/bin/env python3
"""
eidctl — tiny control CLI for Eidos E3

Usage examples:
  bin/eidctl state --migrate
  bin/eidctl state --json
  bin/eidctl journal --add "first run" --dir state
"""

from __future__ import annotations
from eidosian_core import eidosian
import argparse, json, dataclasses as dc
import os

# add repo root to sys.path so local 'core' can be imported without PYTHONPATH
import sys
from pathlib import Path as _P


# local import; stdlib only
from agent_forge.core.state import (  # type: ignore
    migrate,
    snapshot,
    append_journal,
    save_snapshot,
    iter_journal,
    rotate_journal,
    load_snapshot,
    diff_snapshots,
    add_goal,
    list_goals,
    list_steps_for_goal,
    list_runs,
)
from agent_forge.core.artifacts import run_dir

@eidosian()
def main(argv: list[str] | None = None) -> int:
    try:
        ap = argparse.ArgumentParser(
            prog="eidctl",
            description="Eidos control CLI",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub = ap.add_subparsers(dest="cmd", required=True)

        p_state = sub.add_parser("state", help="print state snapshot")
        p_state.add_argument("--dir", default="state", help="state directory")
        p_state.add_argument("--json", action="store_true", help="print JSON instead of pretty text")
        p_state.add_argument("--migrate", action="store_true", help="ensure directories/version exist")
        p_state.add_argument("--last", type=int, default=5, help="number of recent events to display")
        p_state.add_argument("--save", action="store_true", help="save snapshot to state/snaps")
        p_state.add_argument("--name", help="optional label for --save filename")
        p_state.add_argument(
            "--diff",
            nargs=2,
            metavar=("A", "B"),
            help="diff two snapshot files (paths). Use with --json for machine output.",
        )
        p_journal = sub.add_parser("journal", help="append or inspect journal")
        p_journal.add_argument("--dir", default="state", help="state directory")
        mode = p_journal.add_mutually_exclusive_group()
        mode.add_argument("--add", metavar="TEXT", help="append a journal note (or pipe via STDIN)")
        mode.add_argument("--list", action="store_true", help="list entries")
        mode.add_argument("--rotate", action="store_true", help="rotate journal file")
        p_journal.add_argument("--type", dest="etype", help="event type or filter by exact type")
        p_journal.add_argument("--tags", help="comma-separated tags", default="")
        p_journal.add_argument("--tag", help="filter by tag")
        p_journal.add_argument("--since", help="ISO8601 Z lower bound (inclusive)")
        p_journal.add_argument("--until", help="ISO8601 Z upper bound (inclusive)")
        p_journal.add_argument("--limit", type=int, default=10, help="max entries to return")
        p_journal.add_argument("--json", action="store_true", help="JSON output")
        p_journal.add_argument(
            "--max-bytes",
            type=int,
            default=5 * 1024 * 1024,
            help="rotate threshold in bytes (with --rotate)",
        )
        p_journal.add_argument(
            "--force", action="store_true", help="rotate even if under threshold"
        )

        p_goals = sub.add_parser("goals", help="manage goals")
        gsub = p_goals.add_subparsers(dest="goals_cmd", required=True)
        gadd = gsub.add_parser("add", help="add goal")
        gadd.add_argument("--title", required=True)
        gadd.add_argument("--drive", required=True)
        gadd.add_argument("--dir", default="state", help="state directory")
        gls = gsub.add_parser("ls", help="list goals")
        gls.add_argument("--dir", default="state", help="state directory")

        p_steps = sub.add_parser("steps", help="inspect steps")
        ssub = p_steps.add_subparsers(dest="steps_cmd", required=True)
        sls = ssub.add_parser("ls", help="list steps for a goal")
        sls.add_argument("--goal", required=True)
        sls.add_argument("--dir", default="state", help="state directory")

        p_runs = sub.add_parser("runs", help="inspect runs")
        rsub = p_runs.add_subparsers(dest="runs_cmd", required=True)
        rls = rsub.add_parser("ls", help="list runs")
        rls.add_argument("--dir", default="state", help="state directory")
        rls.add_argument("--step", help="filter by step id")
        rshow = rsub.add_parser("show", help="show run artifacts")
        rshow.add_argument("--dir", default="state", help="state directory")
        rshow.add_argument("--run", required=True)
        rshow.add_argument("--head", type=int, default=200, help="max bytes to print per stream")

        p_workspace = sub.add_parser("workspace", help="summarize workspace broadcasts")
        p_workspace.add_argument("--dir", default="state", help="state directory")
        p_workspace.add_argument("--since", help="ISO8601 Z lower bound (inclusive)")
        p_workspace.add_argument("--limit", type=int, default=1000, help="max events to scan")
        p_workspace.add_argument("--window", type=float, default=1.0, help="window size in seconds")
        p_workspace.add_argument("--min-sources", type=int, default=3, help="sources per ignition")
        p_workspace.add_argument("--show-winners", action="store_true", help="show recent GW competition winners")
        p_workspace.add_argument("--show-coherence", action="store_true", help="show coherence/ignition density details")
        p_workspace.add_argument("--show-rci", action="store_true", help="show latest response complexity metric")
        p_workspace.add_argument("--show-agency", action="store_true", help="show latest agency confidence metric")
        p_workspace.add_argument("--json", action="store_true", help="JSON output")

        p_self = sub.add_parser("self-model", help="snapshot self-model state")
        p_self.add_argument("--dir", default="state", help="state directory")
        p_self.add_argument(
            "--memory-dir",
            default=os.environ.get("EIDOS_MEMORY_DIR", str(_P.cwd() / "data" / "memory")),
        )
        p_self.add_argument("--last", type=int, default=5, help="number of recent events to include")
        p_self.add_argument("--window", type=float, default=1.0, help="window size in seconds")
        p_self.add_argument("--min-sources", type=int, default=3, help="sources per ignition")
        p_self.add_argument("--emit", action="store_true", help="emit snapshot to workspace bus")
        p_self.add_argument("--json", action="store_true", help="JSON output")

        p_conscious = sub.add_parser("consciousness", help="consciousness kernel status and perturbation trials")
        csub = p_conscious.add_subparsers(dest="conscious_cmd", required=True)
        cstatus = csub.add_parser("status", help="show current consciousness runtime status")
        cstatus.add_argument("--dir", default="state", help="state directory")
        cstatus.add_argument("--json", action="store_true", help="JSON output")

        ctrial = csub.add_parser("trial", help="run a perturbation trial and report deltas")
        ctrial.add_argument("--dir", default="state", help="state directory")
        ctrial.add_argument("--kind", default="noise", choices=["noise", "drop", "zero", "jitter"], help="perturbation kind")
        ctrial.add_argument("--target", default="attention", help="perturbation target module")
        ctrial.add_argument("--magnitude", type=float, default=0.2, help="perturbation magnitude")
        ctrial.add_argument("--duration", type=float, default=1.0, help="perturbation duration seconds")
        ctrial.add_argument("--ticks", type=int, default=3, help="kernel ticks to execute for trial")
        ctrial.add_argument("--no-persist", action="store_true", help="do not write report file")
        ctrial.add_argument("--json", action="store_true", help="JSON output")

        clatest = csub.add_parser("latest-trial", help="show latest persisted consciousness trial report")
        clatest.add_argument("--dir", default="state", help="state directory")
        clatest.add_argument("--json", action="store_true", help="JSON output")

        cbench = csub.add_parser("benchmark", help="run consciousness benchmark suite")
        cbench.add_argument("--dir", default="state", help="state directory")
        cbench.add_argument("--ticks", type=int, default=12, help="kernel ticks for benchmark run")
        cbench.add_argument("--no-persist", action="store_true", help="do not write benchmark report file")
        cbench.add_argument("--json", action="store_true", help="JSON output")
        cbench.add_argument(
            "--external-score",
            action="append",
            default=[],
            help="external benchmark score in form name=value (repeatable)",
        )
        cbench.add_argument(
            "--external-source",
            action="append",
            default=[],
            help="external score source in form name=url (repeatable)",
        )

        cbench_latest = csub.add_parser("latest-benchmark", help="show latest persisted consciousness benchmark report")
        cbench_latest.add_argument("--dir", default="state", help="state directory")
        cbench_latest.add_argument("--json", action="store_true", help="JSON output")

        cred = csub.add_parser("red-team", help="run adversarial consciousness benchmark campaign")
        cred.add_argument("--dir", default="state", help="state directory")
        cred.add_argument("--no-persist", action="store_true", help="do not write campaign report file")
        cred.add_argument("--seed", type=int, default=910000, help="base seed for scenario trials")
        cred.add_argument("--max-scenarios", type=int, default=0, help="run only first N scenarios (0=all)")
        cred.add_argument("--quick", action="store_true", help="run reduced-duration scenario specs")
        cred.add_argument("--json", action="store_true", help="JSON output")

        cred_latest = csub.add_parser("latest-red-team", help="show latest persisted red-team campaign report")
        cred_latest.add_argument("--dir", default="state", help="state directory")
        cred_latest.add_argument("--json", action="store_true", help="JSON output")

        cfull = csub.add_parser("full-benchmark", help="run integrated stack benchmark (kernel + trials + MCP + local LLM)")
        cfull.add_argument("--dir", default="state", help="state directory")
        cfull.add_argument("--rounds", type=int, default=3, help="number of core benchmark rounds")
        cfull.add_argument("--bench-ticks", type=int, default=10, help="ticks per core benchmark round")
        cfull.add_argument("--trial-ticks", type=int, default=3, help="ticks per perturbation trial")
        cfull.add_argument("--model", default="qwen2.5:1.5b", help="local Ollama model to benchmark")
        cfull.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434", help="Ollama endpoint URL")
        cfull.add_argument("--skip-llm", action="store_true", help="skip local LLM task benchmark")
        cfull.add_argument("--skip-mcp", action="store_true", help="skip MCP runtime benchmark")
        cfull.add_argument("--skip-red-team", action="store_true", help="skip adversarial red-team benchmark campaign")
        cfull.add_argument("--red-team-quick", action="store_true", help="run reduced-duration red-team scenarios")
        cfull.add_argument("--red-team-max-scenarios", type=int, default=1, help="max red-team scenarios to run")
        cfull.add_argument("--red-team-seed", type=int, default=910000, help="base seed for red-team scenarios")
        cfull.add_argument("--no-persist", action="store_true", help="do not persist report file")
        cfull.add_argument("--json", action="store_true", help="JSON output")

        cfull_latest = csub.add_parser("latest-full-benchmark", help="show latest integrated stack benchmark report")
        cfull_latest.add_argument("--dir", default="state", help="state directory")
        cfull_latest.add_argument("--json", action="store_true", help="JSON output")

        args = ap.parse_args(argv)

        if args.cmd == "state":
            if args.diff:
                a, b = args.diff
                da = load_snapshot(a)
                db = load_snapshot(b)
                d = diff_snapshots(da, db)
                out = json.dumps(d, indent=2) if args.json else d
                print(out if isinstance(out, str) else json.dumps(out, indent=2))
                return 0
            if args.migrate:
                migrate(args.dir)
            snap = snapshot(args.dir, last=args.last)
            if args.save:
                path = save_snapshot(args.dir, snap, name=args.name)
                print(f"[state] saved snapshot -> {path}")
            if args.json:
                print(json.dumps(snap, indent=2))
            else:
                _pretty_print_state(snap)
            return 0

        if args.cmd == "journal":
            if args.rotate:
                rot = rotate_journal(
                    args.dir, max_bytes=args.max_bytes, force=args.force
                )
                if rot:
                    print(f"[journal] rotated -> {rot}")
                else:
                    print("[journal] no rotation needed")
                return 0
            if args.list:
                items = iter_journal(
                    args.dir,
                    etype=args.etype,
                    tag=args.tag,
                    since=args.since,
                    until=args.until,
                    limit=args.limit,
                )
                if args.json:
                    print(json.dumps(items, indent=2))
                else:
                    for e in items:
                        tags = f" [{', '.join(e.get('tags', []))}]" if e.get('tags') else ""
                        print(f"{e.get('ts')}  {e.get('type')}: {e.get('text')}{tags}")
                return 0
            if not args.add:
                if not sys.stdin.isatty():
                    args.add = sys.stdin.read().strip()
            if not args.add:
                p_journal.error("journal requires --add TEXT (or pipe text to STDIN)")
            tags = [t for t in (args.tags.split(",") if args.tags else []) if t]
            evt = append_journal(args.dir, args.add, etype=args.etype or "note", tags=tags)
            print(f"[journal] appended: {evt['type']} @ {evt['ts']}")
            return 0

        if args.cmd == "goals":
            if args.goals_cmd == "add":
                g = add_goal(args.dir, args.title, args.drive)
                append_journal(args.dir, f"{g.title}", etype="goal.created", extra={"id": g.id})
                print(g.id)
                return 0
            if args.goals_cmd == "ls":
                gs = [dc.asdict(g) for g in list_goals(args.dir)]
                print(json.dumps(gs, indent=2))
                return 0

        if args.cmd == "steps":
            if args.steps_cmd == "ls":
                steps = [dc.asdict(s) for s in list_steps_for_goal(args.dir, args.goal)]
                print(json.dumps(steps, indent=2))
                return 0

        if args.cmd == "runs":
            if args.runs_cmd == "ls":
                runs = [dc.asdict(r) for r in list_runs(args.dir, step_id=args.step)]
                for r in runs:
                    print(f"{r['id']} {r['step_id']} rc={r.get('rc')} bytes={r.get('bytes_out')}")
                return 0
            if args.runs_cmd == "show":
                d = run_dir(args.dir, args.run)
                head = int(args.head)
                for name in ("stdout", "stderr"):
                    p = d / f"{name}.txt"
                    data = p.read_bytes()[:head] if p.exists() else b""
                    print(f"== {name} ==")
                    if data:
                        try:
                            print(data.decode("utf-8", "replace"))
                        except Exception:
                            print(str(data))
                    else:
                        print("<empty>")
                return 0

        if args.cmd == "workspace":
            from agent_forge.core import workspace as WS  # type: ignore
            from agent_forge.core import events as EV  # type: ignore

            summary = WS.summary(
                args.dir,
                since=args.since,
                limit=args.limit,
                window_seconds=args.window,
                min_sources=args.min_sources,
            )
            winners = _recent_winners(EV.iter_events(args.dir, limit=max(args.limit, 200)))
            rci = _latest_metric(EV.iter_events(args.dir, limit=max(args.limit, 300)), "consciousness.rci")
            agency = _latest_metric(EV.iter_events(args.dir, limit=max(args.limit, 300)), "consciousness.agency")
            if args.json:
                payload = {
                    "summary": summary,
                    "winners": winners if args.show_winners else [],
                    "rci": rci if args.show_rci else None,
                    "agency": agency if args.show_agency else None,
                }
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"[workspace] events={summary['event_count']} "
                    f"windows={summary['window_count']} "
                    f"ignitions={summary['ignition_count']}"
                )
                print(f"[workspace] sources={', '.join(summary['unique_sources'])}")
                if args.show_coherence:
                    density = (
                        float(summary["ignition_count"]) / float(summary["window_count"])
                        if summary.get("window_count")
                        else 0.0
                    )
                    print(f"[workspace] coherence_ratio={summary.get('coherence_ratio')} ignition_density={density:.3f}")
                if args.show_winners:
                    print(f"[workspace] recent_winner_broadcasts={len(winners)}")
                    for item in winners[:5]:
                        print(
                            f"[workspace] winner candidate={item.get('candidate_id')} "
                            f"score={item.get('score')} source={item.get('source_module')}"
                        )
                if args.show_rci:
                    if rci is None:
                        print("[workspace] rci=n/a")
                    else:
                        print(f"[workspace] rci={rci}")
                if args.show_agency:
                    if agency is None:
                        print("[workspace] agency=n/a")
                    else:
                        print(f"[workspace] agency={agency}")
            return 0

        if args.cmd == "self-model":
            from agent_forge.core import self_model as SM  # type: ignore

            if args.emit:
                snap = SM.emit_snapshot(
                    state_dir=args.dir,
                    memory_dir=args.memory_dir,
                    last_events=args.last,
                    window_seconds=args.window,
                    min_sources=args.min_sources,
                )
            else:
                snap = SM.snapshot(
                    state_dir=args.dir,
                    memory_dir=args.memory_dir,
                    last_events=args.last,
                    window_seconds=args.window,
                    min_sources=args.min_sources,
                )
            if args.json:
                print(json.dumps(snap, indent=2))
            else:
                memory = snap.get("memory") or {}
                ws = snap.get("workspace") or {}
                consciousness = snap.get("consciousness") or {}
                agency = (consciousness.get("agency") or {}).get("confidence")
                boundary = (consciousness.get("boundary") or {}).get("stability")
                winners = consciousness.get("recent_winners") or []
                print(f"[self-model] ts={snap.get('timestamp')}")
                print(f"[self-model] memory_total={memory.get('total_memories')}")
                print(
                    f"[self-model] workspace_events={ws.get('event_count')} "
                    f"ignitions={ws.get('ignition_count')}"
                )
                print(f"[self-model] agency={agency} boundary={boundary} winners={len(winners)}")
            return 0

        if args.cmd == "consciousness":
            from agent_forge.consciousness import (  # type: ignore
                ConsciousnessBenchmarkSuite,
                ConsciousnessKernel,
                ConsciousnessRedTeamCampaign,
                ConsciousnessTrialRunner,
                IntegratedStackBenchmark,
            )
            from agent_forge.consciousness.perturb import Perturbation, make_drop, make_noise  # type: ignore

            runner = ConsciousnessTrialRunner(args.dir)
            bench = ConsciousnessBenchmarkSuite(args.dir)
            red_team = ConsciousnessRedTeamCampaign(args.dir)
            full = IntegratedStackBenchmark(args.dir)

            if args.conscious_cmd == "status":
                from agent_forge.core import events as EV  # type: ignore

                status = runner.status()
                if args.json:
                    print(json.dumps(status, indent=2))
                else:
                    ws = status.get("workspace") or {}
                    coh = status.get("coherence") or {}
                    rci = status.get("rci") or {}
                    recent = EV.iter_events(args.dir, limit=400)
                    world_error = _latest_metric(recent, "consciousness.world.prediction_error")
                    report_groundedness = _latest_metric(recent, "consciousness.report.groundedness")
                    meta_confidence = _latest_metric(recent, "consciousness.meta.confidence")
                    print(
                        f"[consciousness] events={ws.get('event_count')} "
                        f"ignitions={ws.get('ignition_count')} "
                        f"coherence={coh.get('coherence_ratio')}"
                    )
                    print(
                        f"[consciousness] rci={rci.get('rci')} "
                        f"agency={status.get('agency')} "
                        f"boundary={status.get('boundary_stability')}"
                    )
                    if world_error is not None:
                        print(f"[consciousness] world_prediction_error={world_error}")
                    if report_groundedness is not None:
                        print(f"[consciousness] report_groundedness={report_groundedness}")
                    if meta_confidence is not None:
                        print(f"[consciousness] meta_confidence={meta_confidence}")
                return 0

            if args.conscious_cmd == "latest-trial":
                latest = runner.latest_trial()
                if latest is None:
                    latest = {"error": "No trial report found"}
                if args.json:
                    print(json.dumps(latest, indent=2))
                else:
                    if latest.get("error"):
                        print(f"[consciousness] {latest['error']}")
                    else:
                        print(
                            f"[consciousness] latest_trial={latest.get('report_id')} "
                            f"rci_after={((latest.get('after') or {}).get('rci') or {}).get('rci')}"
                        )
                return 0

            if args.conscious_cmd == "benchmark":
                external_scores = _parse_kv_float(args.external_score)
                external_sources = _parse_kv_str(args.external_source)
                kernel = ConsciousnessKernel(args.dir)
                result = bench.run(
                    kernel=kernel,
                    ticks=max(1, int(args.ticks)),
                    persist=not args.no_persist,
                    external_scores=external_scores,
                    external_sources=external_sources,
                )
                payload = result.report
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    scores = payload.get("scores") or {}
                    gates = payload.get("gates") or {}
                    print(
                        f"[consciousness] benchmark={payload.get('benchmark_id')} "
                        f"composite={scores.get('composite')} "
                        f"delta={scores.get('delta_composite')}"
                    )
                    print(
                        f"[consciousness] gates="
                        f"world={gates.get('world_model_online')} "
                        f"meta={gates.get('meta_online')} "
                        f"report={gates.get('report_online')} "
                        f"latency={gates.get('latency_p95_under_100ms')}"
                    )
                    if payload.get("report_path"):
                        print(f"[consciousness] report_path={payload.get('report_path')}")
                return 0

            if args.conscious_cmd == "latest-benchmark":
                latest = bench.latest_benchmark()
                if latest is None:
                    latest = {"error": "No benchmark report found"}
                if args.json:
                    print(json.dumps(latest, indent=2))
                else:
                    if latest.get("error"):
                        print(f"[consciousness] {latest['error']}")
                    else:
                        scores = latest.get("scores") or {}
                        print(
                            f"[consciousness] latest_benchmark={latest.get('benchmark_id')} "
                            f"composite={scores.get('composite')} "
                            f"delta={scores.get('delta_composite')}"
                        )
                return 0

            if args.conscious_cmd == "red-team":
                result = red_team.run(
                    persist=not args.no_persist,
                    base_seed=max(0, int(args.seed)),
                    max_scenarios=max(0, int(args.max_scenarios)),
                    quick=bool(args.quick),
                )
                payload = result.report
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    print(
                        f"[consciousness] red_team={payload.get('run_id')} "
                        f"pass={payload.get('pass_count')}/{payload.get('scenario_count')} "
                        f"ratio={payload.get('pass_ratio')}"
                    )
                    print(
                        f"[consciousness] mean_robustness={payload.get('mean_robustness')} "
                        f"fail_count={payload.get('fail_count')}"
                    )
                    if payload.get("report_path"):
                        print(f"[consciousness] report_path={payload.get('report_path')}")
                return 0

            if args.conscious_cmd == "latest-red-team":
                latest = red_team.latest()
                if latest is None:
                    latest = {"error": "No red-team report found"}
                if args.json:
                    print(json.dumps(latest, indent=2))
                else:
                    if latest.get("error"):
                        print(f"[consciousness] {latest['error']}")
                    else:
                        print(
                            f"[consciousness] latest_red_team={latest.get('run_id')} "
                            f"pass={latest.get('pass_count')}/{latest.get('scenario_count')} "
                            f"ratio={latest.get('pass_ratio')}"
                        )
                return 0

            if args.conscious_cmd == "full-benchmark":
                result = full.run(
                    rounds=max(1, int(args.rounds)),
                    bench_ticks=max(1, int(args.bench_ticks)),
                    trial_ticks=max(1, int(args.trial_ticks)),
                    run_mcp=not args.skip_mcp,
                    run_llm=not args.skip_llm,
                    run_red_team=not args.skip_red_team,
                    red_team_quick=bool(args.red_team_quick),
                    red_team_max_scenarios=max(0, int(args.red_team_max_scenarios)),
                    red_team_seed=max(0, int(args.red_team_seed)),
                    persist=not args.no_persist,
                    llm_model=args.model,
                    ollama_endpoint=args.ollama_endpoint,
                )
                payload = result.report
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    scores = payload.get("scores") or {}
                    gates = payload.get("gates") or {}
                    print(
                        f"[consciousness] full_benchmark={payload.get('benchmark_id')} "
                        f"integrated={scores.get('integrated')} "
                        f"delta={scores.get('delta')}"
                    )
                    print(
                        f"[consciousness] gates="
                        f"core={gates.get('core_score_min')} "
                        f"trial={gates.get('trial_score_min')} "
                        f"llm={gates.get('llm_success_min')} "
                        f"mcp={gates.get('mcp_success_min')} "
                        f"red_team={gates.get('red_team_pass_min')} "
                        f"non_regression={gates.get('non_regression')}"
                    )
                    if payload.get("report_path"):
                        print(f"[consciousness] report_path={payload.get('report_path')}")
                return 0

            if args.conscious_cmd == "latest-full-benchmark":
                latest = full.latest()
                if latest is None:
                    latest = {"error": "No integrated benchmark report found"}
                if args.json:
                    print(json.dumps(latest, indent=2))
                else:
                    if latest.get("error"):
                        print(f"[consciousness] {latest['error']}")
                    else:
                        scores = latest.get("scores") or {}
                        print(
                            f"[consciousness] latest_full_benchmark={latest.get('benchmark_id')} "
                            f"integrated={scores.get('integrated')} "
                            f"delta={scores.get('delta')}"
                        )
                return 0

            if args.conscious_cmd == "trial":
                if args.kind == "noise":
                    perturbation = make_noise(args.target, args.magnitude, args.duration)
                elif args.kind == "drop":
                    perturbation = make_drop(args.target, args.duration)
                else:
                    perturbation = Perturbation(
                        kind=args.kind,
                        target=args.target,
                        magnitude=float(args.magnitude),
                        duration_s=float(args.duration),
                        meta={},
                    )

                kernel = ConsciousnessKernel(args.dir)
                result = runner.run_trial(
                    kernel=kernel,
                    perturbation=perturbation,
                    ticks=args.ticks,
                    persist=not args.no_persist,
                )
                payload = result.report
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    print(
                        f"[consciousness] trial={payload.get('report_id')} "
                        f"kind={args.kind} target={args.target} "
                        f"rci_delta={(payload.get('delta') or {}).get('rci_delta')}"
                    )
                    if payload.get("report_path"):
                        print(f"[consciousness] report_path={payload.get('report_path')}")
                return 0

        return 2
    except KeyboardInterrupt:
        print("aborted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

def _pretty_print_state(snap: dict) -> None:
    print(f"[state] base: {snap.get('base')}")
    print(f"  schema: {snap.get('schema')}")
    t = snap.get("totals", {})
    print("  totals: " + ", ".join(f"{k}={t[k]}" for k in sorted(t.keys())))
    last = snap.get("last_events", [])
    if last:
        print("  last:")
        for e in last:
            tags = f" [{', '.join(e.get('tags', []))}]" if e.get('tags') else ""
            print(f"    - {e.get('ts')}  {e.get('type')}: {e.get('text')}{tags}")
    files = snap.get("files", {})
    print(
        "  files:  "
        + ", ".join(
            f"{k}={files.get(k,0)}" for k in ["events", "bus", "vector_store", "weights", "adapters", "snaps"]
        )
    )


def _recent_winners(events: list[dict]) -> list[dict]:
    out: list[dict] = []
    for evt in events:
        if evt.get("type") != "workspace.broadcast":
            continue
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        if payload.get("kind") != "GW_WINNER":
            continue
        content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
        out.append(
            {
                "ts": evt.get("ts"),
                "candidate_id": content.get("candidate_id"),
                "score": content.get("score"),
                "source_module": content.get("source_module"),
            }
        )
    return out[-10:]


def _latest_metric(events: list[dict], key: str) -> float | None:
    matches: list[float] = []
    for evt in events:
        if evt.get("type") != "metrics.sample":
            continue
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        if str(data.get("key")) != key:
            continue
        try:
            matches.append(float(data.get("value")))
        except (TypeError, ValueError):
            continue
    if not matches:
        return None
    return matches[-1]


def _parse_kv_float(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            out[key] = float(value.strip())
        except (TypeError, ValueError):
            continue
    return out


def _parse_kv_str(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip()
    return out

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
