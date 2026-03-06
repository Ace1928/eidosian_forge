from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from eidosian_core import eidosian

from agent_forge.core import events as BUS
from agent_forge.core import state as S
from agent_forge.core.scheduler import create_plan_for_goal

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _ensure_bridge_import_path() -> None:
    root = _forge_root()
    for path in (
        root / "lib",
        root / "memory_forge" / "src",
        root / "knowledge_forge" / "src",
        root / "eidos_mcp" / "src",
        root / "ollama_forge" / "src",
        root,
    ):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


@eidosian()
def load_supervisor_config(path: str | Path | None) -> dict[str, Any]:
    if path is None or yaml is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class AutonomySupervisor:
    """Policy-driven mission seeding for the long-running daemon."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        repo_root: str | Path = ".",
        config: Mapping[str, Any] | None = None,
        bridge: Any = None,
        memory_system: Any = None,
        graphrag: Any = None,
        embedder: Any = None,
    ) -> None:
        self.state_dir = str(state_dir)
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config = dict(config or {})
        self._bridge = bridge
        self._memory_system = memory_system
        self._graphrag = graphrag
        self._embedder = embedder

    def _policy(self) -> dict[str, Any]:
        policy = self.config.get("policy", {})
        return dict(policy) if isinstance(policy, Mapping) else {}

    def _missions(self) -> list[dict[str, Any]]:
        missions = self.config.get("missions", [])
        if not isinstance(missions, list):
            return []
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(missions):
            if not isinstance(item, Mapping):
                continue
            mission = dict(item)
            mission.setdefault("id", f"mission_{idx + 1}")
            mission.setdefault("enabled", True)
            mission.setdefault("priority", 0.5)
            mission.setdefault("cooldown_beats", 0)
            mission.setdefault("max_runs", 0)
            mission.setdefault("vars", {})
            out.append(mission)
        return out

    def _load_bridge(self) -> tuple[Any, str]:
        if self._bridge is not None:
            return self._bridge, ""
        try:
            _ensure_bridge_import_path()
            from eidosian_vector import build_default_embedder  # type: ignore

            from knowledge_forge import KnowledgeMemoryBridge  # type: ignore

            if self._embedder is None:
                self._embedder = build_default_embedder()
            self._bridge = KnowledgeMemoryBridge(embedder=self._embedder)
            return self._bridge, ""
        except Exception as exc:  # pragma: no cover - defensive fallback
            return None, str(exc)

    def _load_memory_system(self) -> Any:
        if self._memory_system is not None:
            return self._memory_system
        try:
            _ensure_bridge_import_path()
            from eidosian_vector import build_default_embedder  # type: ignore

            from memory_forge import TieredMemorySystem  # type: ignore

            if self._embedder is None:
                self._embedder = build_default_embedder()
            self._memory_system = TieredMemorySystem(embedder=self._embedder)
        except Exception:  # pragma: no cover - defensive fallback
            self._memory_system = None
        return self._memory_system

    def _load_graphrag(self) -> Any:
        if self._graphrag is not None:
            return self._graphrag
        try:
            _ensure_bridge_import_path()
            from knowledge_forge.integrations.graphrag import GraphRAGIntegration  # type: ignore

            root = self.repo_root / "graphrag_workspace"
            if not root.exists():
                root = self.repo_root / "graphrag"
            self._graphrag = GraphRAGIntegration(graphrag_root=root)
        except Exception:  # pragma: no cover - defensive fallback
            self._graphrag = None
        return self._graphrag

    def _runtime_status(self) -> dict[str, Any]:
        runtime_dir = self.repo_root / "data" / "runtime"
        pipeline = _read_json_dict(runtime_dir / "living_pipeline_status.json")
        scheduler = _read_json_dict(runtime_dir / "eidos_scheduler_status.json")
        coordinator = _read_json_dict(runtime_dir / "forge_coordinator_status.json")
        return {
            "pipeline": pipeline,
            "scheduler": scheduler,
            "coordinator": coordinator,
        }

    def _recent_context_query(self) -> str:
        parts: list[str] = []
        base_query = _to_text(self.config.get("context_query"))
        if base_query:
            parts.append(base_query)
        for event_type in ("memory_bridge.status", "knowledge_bridge.status"):
            for evt in BUS.iter_events(self.state_dir, limit=80):
                if evt.get("type") != event_type:
                    continue
                data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
                parts.append(_to_text(data.get("query")))
        if not parts:
            parts.append("consciousness benchmark latency memory knowledge autonomy")
        tokens: list[str] = []
        seen: set[str] = set()
        for part in parts:
            for token in _TOKEN_RE.findall(part):
                lowered = token.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                tokens.append(lowered)
                if len(tokens) >= int(self.config.get("query_max_tokens", 24) or 24):
                    return " ".join(tokens)
        return " ".join(tokens)

    def _context_packet(self) -> dict[str, Any]:
        limit = max(1, int(self.config.get("context_limit", 6) or 6))
        query = self._recent_context_query()
        bridge, load_error = self._load_bridge()
        results = []
        if bridge is not None:
            try:
                results = bridge.unified_search(query, limit=limit)
            except Exception as exc:  # pragma: no cover - defensive fallback
                load_error = str(exc)
                results = []

        context_tokens: set[str] = set()
        hits: list[dict[str, Any]] = []
        for result in results:
            metadata = dict(getattr(result, "metadata", {}) or {})
            tags = [str(tag) for tag in metadata.get("tags", [])]
            content = _to_text(getattr(result, "content", ""))
            context_tokens |= _tokenize(" ".join([content, " ".join(tags)]))
            hits.append(
                {
                    "source": _to_text(getattr(result, "source", "")),
                    "id": _to_text(getattr(result, "id", "")),
                    "score": float(getattr(result, "score", 0.0) or 0.0),
                    "content": content[:240],
                    "tags": tags[:8],
                }
            )

        report_summary: dict[str, Any] = {}
        artifact_summary: dict[str, Any] = {}
        trend_summary: dict[str, Any] = {}
        graphrag = self._load_graphrag()
        if graphrag is not None:
            try:
                report_summary = graphrag.native_report_summary(limit=4) or {}
            except Exception:  # pragma: no cover - defensive fallback
                report_summary = {}
            try:
                artifact_summary = graphrag.native_artifact_summary(limit=6) or {}
            except Exception:  # pragma: no cover - defensive fallback
                artifact_summary = {}
            try:
                trend_summary = graphrag.native_trend_summary(limit=8) or {}
            except Exception:  # pragma: no cover - defensive fallback
                trend_summary = {}

        for row in report_summary.get("reports") or []:
            if not isinstance(row, Mapping):
                continue
            context_tokens |= _tokenize(
                " ".join(
                    [
                        _to_text(row.get("community")),
                        _to_text(row.get("title")),
                        _to_text(row.get("summary")),
                        _to_text(row.get("quality_band")),
                    ]
                )
            )
        for row in artifact_summary.get("artifacts") or []:
            if not isinstance(row, Mapping):
                continue
            parts = [
                _to_text(row.get("artifact_path")),
                _to_text(row.get("kind")),
            ]
            if row.get("benchmark_gate_pass") is False:
                parts.append("benchmark failure gate fail")
            if int(row.get("drift_warning_count") or 0) > 0:
                parts.append("drift warnings")
            context_tokens |= _tokenize(" ".join(parts))
        latest_trend = trend_summary.get("latest") if isinstance(trend_summary.get("latest"), Mapping) else {}
        if latest_trend:
            context_tokens |= _tokenize(
                " ".join(
                    [
                        " ".join(str(x) for x in latest_trend.get("weak_community_labels") or []),
                        " ".join(str(x) for x in (latest_trend.get("artifact_kinds") or {}).keys()),
                    ]
                )
            )

        runtime = self._runtime_status()
        scheduler = runtime.get("scheduler") if isinstance(runtime.get("scheduler"), Mapping) else {}
        pipeline = runtime.get("pipeline") if isinstance(runtime.get("pipeline"), Mapping) else {}
        coordinator = runtime.get("coordinator") if isinstance(runtime.get("coordinator"), Mapping) else {}
        runtime_parts = [
            _to_text(scheduler.get("state")),
            _to_text(scheduler.get("current_task")),
            _to_text(pipeline.get("state")),
            _to_text(pipeline.get("phase")),
            _to_text(coordinator.get("state")),
            _to_text(coordinator.get("task")),
            " ".join(_to_text(item.get("model")) for item in (coordinator.get("active_models") or []) if isinstance(item, Mapping)),
        ]
        context_tokens |= _tokenize(" ".join(part for part in runtime_parts if part))

        return {
            "query": query,
            "load_error": load_error,
            "hits": hits,
            "tokens": sorted(context_tokens),
            "report_summary": report_summary,
            "artifact_summary": artifact_summary,
            "trend_summary": trend_summary,
            "runtime": runtime,
        }

    def _recent_selections(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for evt in BUS.iter_events(self.state_dir, limit=400):
            if evt.get("type") != "autonomy.mission_selected":
                continue
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            rows.append(dict(data))
        return rows

    def _active_goals(self) -> set[str]:
        plan_by_goal = {plan.id: plan.goal_id for plan in S.list_plans(self.state_dir)}
        active_goal_ids: set[str] = set()
        for step in S.list_steps(self.state_dir):
            if step.status in {"todo", "running"}:
                goal_id = plan_by_goal.get(step.plan_id)
                if goal_id:
                    active_goal_ids.add(goal_id)
        return active_goal_ids

    def _repo_dirty(self) -> bool:
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return False
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def _mission_history(self, mission_id: str) -> tuple[int, int]:
        count = 0
        last_beat = -10_000
        for row in self._recent_selections():
            if row.get("mission_id") != mission_id:
                continue
            count += 1
            try:
                last_beat = max(last_beat, int(row.get("beat", -10_000)))
            except (TypeError, ValueError):
                continue
        return count, last_beat

    def _score_mission(self, mission: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        base = float(mission.get("priority", 0.5) or 0.5)
        query_tokens = _tokenize(_to_text(mission.get("query")) or _to_text(mission.get("title")))
        if not query_tokens:
            return base
        context_tokens = set(context.get("tokens") or [])
        overlap = len(query_tokens & context_tokens)
        score = base + (overlap / max(len(query_tokens), 1))

        template = _to_text(mission.get("template")).lower()
        report_summary = context.get("report_summary") if isinstance(context.get("report_summary"), Mapping) else {}
        artifact_summary = (
            context.get("artifact_summary") if isinstance(context.get("artifact_summary"), Mapping) else {}
        )
        trend_summary = context.get("trend_summary") if isinstance(context.get("trend_summary"), Mapping) else {}
        avg_quality = float(report_summary.get("average_quality_score") or 0.0)
        weak_communities = int(report_summary.get("weak_communities") or 0)
        benchmark_failures = int(artifact_summary.get("benchmark_failures") or 0)
        drift_warning_artifacts = int(artifact_summary.get("drift_warning_artifacts") or 0)
        latest_trend = trend_summary.get("latest") if isinstance(trend_summary.get("latest"), Mapping) else {}
        weak_labels = {
            _to_text(item).lower().replace(" ", "_")
            for item in latest_trend.get("weak_community_labels") or []
            if _to_text(item)
        }
        artifact_kinds = {
            _to_text(item).lower()
            for item in (latest_trend.get("artifact_kinds") or {}).keys()
            if _to_text(item)
        }
        focus_communities = {
            _to_text(item).lower().replace(" ", "_")
            for item in mission.get("focus_communities", []) or []
            if _to_text(item)
        }
        focus_artifact_kinds = {
            _to_text(item).lower()
            for item in mission.get("focus_artifact_kinds", []) or []
            if _to_text(item)
        }
        targeted_overlap = len(focus_communities & weak_labels) + len(focus_artifact_kinds & artifact_kinds)
        runtime = context.get("runtime") if isinstance(context.get("runtime"), Mapping) else {}
        scheduler = runtime.get("scheduler") if isinstance(runtime.get("scheduler"), Mapping) else {}
        pipeline = runtime.get("pipeline") if isinstance(runtime.get("pipeline"), Mapping) else {}
        coordinator = runtime.get("coordinator") if isinstance(runtime.get("coordinator"), Mapping) else {}
        scheduler_state = _to_text(scheduler.get("state")).lower()
        pipeline_state = _to_text(pipeline.get("state")).lower()
        pipeline_phase = _to_text(pipeline.get("phase")).lower()
        coordinator_state = _to_text(coordinator.get("state")).lower()
        active_models = list(coordinator.get("active_models") or [])
        is_model_heavy = template in {"consciousness_guard"} or bool(mission.get("requires_llm", False))
        if scheduler_state == "error" or pipeline_state == "failed" or coordinator_state == "error":
            score += 0.5 if template == "hygiene" else 0.1
        if pipeline_phase in {"graphrag", "living_documentation", "word_forge"} and is_model_heavy and active_models:
            score -= 0.35
        if scheduler_state == "running" and template == "hygiene":
            score += 0.15

        if template == "hygiene":
            score += min(1.4, benchmark_failures * 0.45 + drift_warning_artifacts * 0.25)
            if avg_quality and avg_quality < 0.75:
                score += min(0.45, 0.75 - avg_quality)
        elif template == "consciousness_guard":
            if avg_quality >= 0.7:
                score += min(0.25, avg_quality * 0.25)
            score -= min(0.2, weak_communities * 0.05)
        elif template == "lint":
            score += min(0.4, benchmark_failures * 0.2)

        if targeted_overlap:
            score += min(0.6, targeted_overlap * 0.15)

        return score

    def _remember_selection(self, mission: Mapping[str, Any], context: Mapping[str, Any], score: float) -> None:
        memory = self._load_memory_system()
        if memory is None or not hasattr(memory, "remember"):
            return
        content = (
            f"Autonomy supervisor selected mission {mission.get('id')}: "
            f"{mission.get('title')} with score {score:.3f}. "
            f"Context query: {context.get('query')}"
        )
        metadata = {
            "mission_id": _to_text(mission.get("id")),
            "template": _to_text(mission.get("template")),
            "context_query": _to_text(context.get("query")),
        }
        try:
            _ensure_bridge_import_path()
            from memory_forge.core.interfaces import MemoryType  # type: ignore
            from memory_forge.core.tiered_memory import MemoryNamespace, MemoryTier  # type: ignore

            memory.remember(
                content,
                tier=MemoryTier.WORKING,
                namespace=MemoryNamespace.TASK,
                memory_type=MemoryType.EPISODIC,
                importance=0.9,
                tags={"autonomy", "supervisor", str(mission.get("id"))},
                metadata=metadata,
            )
        except Exception:
            try:
                memory.remember(content, importance=0.9, metadata=metadata)
            except Exception:
                return

    @eidosian()
    def tick(self, *, beat_count: int) -> dict[str, Any]:
        active_goal_ids = self._active_goals()
        active_goals = [goal for goal in S.list_goals(self.state_dir) if goal.id in active_goal_ids]
        policy = self._policy()
        max_active_goals = max(1, int(policy.get("max_active_goals", 1) or 1))
        allowed_templates = {str(x) for x in policy.get("allowed_templates", []) if x}
        require_clean_git = bool(policy.get("require_clean_git", False))
        repo_dirty = self._repo_dirty()
        context = self._context_packet()

        ctx_payload = {
            "beat": int(beat_count),
            "query": context.get("query"),
            "hit_count": len(context.get("hits") or []),
            "report_count": int(((context.get("report_summary") or {}).get("count")) or 0),
            "artifact_count": int(((context.get("artifact_summary") or {}).get("count")) or 0),
            "average_report_quality": float(((context.get("report_summary") or {}).get("average_quality_score")) or 0.0),
            "benchmark_failures": int(((context.get("artifact_summary") or {}).get("benchmark_failures")) or 0),
            "weak_communities": list((((context.get("trend_summary") or {}).get("latest")) or {}).get("weak_community_labels") or []),
            "artifact_kinds": sorted(list(((((context.get("trend_summary") or {}).get("latest")) or {}).get("artifact_kinds") or {}).keys())),
            "repo_root": str(self.repo_root),
            "repo_dirty": repo_dirty,
            "scheduler_state": _to_text((((context.get("runtime") or {}).get("scheduler")) or {}).get("state")),
            "pipeline_state": _to_text((((context.get("runtime") or {}).get("pipeline")) or {}).get("state")),
            "pipeline_phase": _to_text((((context.get("runtime") or {}).get("pipeline")) or {}).get("phase")),
            "coordinator_state": _to_text((((context.get("runtime") or {}).get("coordinator")) or {}).get("state")),
            "active_model_count": len(((((context.get("runtime") or {}).get("coordinator")) or {}).get("active_models")) or []),
        }
        BUS.append(self.state_dir, "autonomy.context", ctx_payload, tags=["autonomy", "context"])

        if len(active_goals) >= max_active_goals:
            payload = {
                "status": "busy",
                "beat": int(beat_count),
                "active_goals": [goal.title for goal in active_goals],
            }
            BUS.append(self.state_dir, "autonomy.idle", payload, tags=["autonomy", "idle"])
            return payload

        candidates: list[tuple[float, dict[str, Any]]] = []
        goal_titles = {goal.title for goal in S.list_goals(self.state_dir)}
        for mission in self._missions():
            if not bool(mission.get("enabled", True)):
                continue
            mission_id = _to_text(mission.get("id"))
            template = _to_text(mission.get("template"))
            title = _to_text(mission.get("title"))
            if not template or not title:
                continue

            blocked_reason = ""
            if allowed_templates and template not in allowed_templates:
                blocked_reason = "template_not_allowed"
            elif require_clean_git and repo_dirty:
                blocked_reason = "repo_dirty"
            elif title in goal_titles:
                blocked_reason = "goal_exists"
            else:
                count, last_beat = self._mission_history(mission_id)
                max_runs = int(mission.get("max_runs", 0) or 0)
                cooldown = int(mission.get("cooldown_beats", 0) or 0)
                if max_runs and count >= max_runs:
                    blocked_reason = "max_runs_reached"
                elif cooldown and (beat_count - last_beat) < cooldown:
                    blocked_reason = "cooldown_active"

            if blocked_reason:
                BUS.append(
                    self.state_dir,
                    "autonomy.mission_blocked",
                    {
                        "beat": int(beat_count),
                        "mission_id": mission_id,
                        "template": template,
                        "reason": blocked_reason,
                    },
                    tags=["autonomy", "blocked"],
                )
                continue

            candidates.append((self._score_mission(mission, context), mission))

        if not candidates:
            payload = {"status": "idle", "beat": int(beat_count), "reason": "no_eligible_missions"}
            BUS.append(self.state_dir, "autonomy.idle", payload, tags=["autonomy", "idle"])
            return payload

        score, mission = max(candidates, key=lambda item: item[0])
        goal = S.add_goal(self.state_dir, _to_text(mission["title"]), _to_text(mission.get("drive", "integrity")))
        plan = create_plan_for_goal(
            self.state_dir,
            goal,
            template=_to_text(mission["template"]),
            vars=mission.get("vars") if isinstance(mission.get("vars"), Mapping) else {},
            meta={
                "cwd": str(self.repo_root),
                "mission_id": _to_text(mission.get("id")),
                "priority": float(mission.get("priority", 0.5) or 0.5),
                "context_query": _to_text(context.get("query")),
                "created_by": "autonomy_supervisor",
            },
        )
        payload = {
            "status": "selected",
            "beat": int(beat_count),
            "mission_id": _to_text(mission.get("id")),
            "goal_id": goal.id,
            "plan_id": plan.id,
            "score": round(score, 4),
            "template": _to_text(mission.get("template")),
            "context_query": _to_text(context.get("query")),
            "context_hits": len(context.get("hits") or []),
            "report_count": int(((context.get("report_summary") or {}).get("count")) or 0),
            "artifact_count": int(((context.get("artifact_summary") or {}).get("count")) or 0),
            "average_report_quality": float(((context.get("report_summary") or {}).get("average_quality_score")) or 0.0),
            "benchmark_failures": int(((context.get("artifact_summary") or {}).get("benchmark_failures")) or 0),
            "weak_communities": list((((context.get("trend_summary") or {}).get("latest")) or {}).get("weak_community_labels") or []),
            "artifact_kinds": sorted(list(((((context.get("trend_summary") or {}).get("latest")) or {}).get("artifact_kinds") or {}).keys())),
            "scheduler_state": _to_text((((context.get("runtime") or {}).get("scheduler")) or {}).get("state")),
            "pipeline_phase": _to_text((((context.get("runtime") or {}).get("pipeline")) or {}).get("phase")),
            "active_model_count": len(((((context.get("runtime") or {}).get("coordinator")) or {}).get("active_models")) or []),
        }
        BUS.append(self.state_dir, "autonomy.mission_selected", payload, tags=["autonomy", "selected"])
        S.append_journal(
            self.state_dir,
            json.dumps(payload, sort_keys=True),
            etype="autonomy.mission_selected",
            tags=["autonomy", _to_text(mission.get("id"))],
        )
        self._remember_selection(mission, context, score)
        return payload
