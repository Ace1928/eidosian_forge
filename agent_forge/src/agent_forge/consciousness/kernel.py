from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agent_forge.core import events as bus
from agent_forge.core import workspace

from .modules.attention import AttentionModule
from .modules.affect import AffectModule
from .modules.autotune import AutotuneModule
from .modules.intero import InteroModule
from .modules.knowledge_bridge import KnowledgeBridgeModule
from .modules.memory_bridge import MemoryBridgeModule
from .modules.meta import MetaModule
from .modules.policy import PolicyModule
from .modules.phenomenology_probe import PhenomenologyProbeModule
from .modules.report import ReportModule
from .modules.sense import SenseModule
from .modules.self_model_ext import SelfModelExtModule
from .modules.simulation import SimulationModule
from .modules.world_model import WorldModelModule
from .modules.working_set import WorkingSetModule
from .modules.workspace_competition import WorkspaceCompetitionModule
from .state_store import ModuleStateStore
from .tuning.overlay import load_tuned_overlay, resolve_config
from .types import Module, TickContext, merged_config


@dataclass
class KernelResult:
    ts: str
    emitted_events: int
    modules: List[str]
    errors: List[str]


class ConsciousnessKernel:
    def __init__(
        self,
        state_dir: str | Path,
        *,
        config: Optional[Mapping[str, Any]] = None,
        modules: Optional[Iterable[Module]] = None,
        seed: int = 1337,
        respect_tuned_overlay: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir)
        self._base_config = merged_config(config or {})
        self._runtime_overrides: Dict[str, Any] = {}
        self._respect_tuned_overlay = bool(respect_tuned_overlay)
        self._tuned_overlay: Dict[str, Any] = {}
        self._last_invalid_overlay_keys: tuple[str, ...] = ()
        self.config = dict(self._base_config)
        self.rng = random.Random(seed)
        self.state_store = ModuleStateStore(
            self.state_dir,
            autosave_interval_secs=float(
                self._base_config.get("state_autosave_interval_secs", 2.0)
            ),
        )
        self._refresh_config()
        self.beat_count = int(self.state_store.get_meta("beat_count", 0) or 0)
        self._active_perturbations: list[dict[str, Any]] = []
        self.modules: List[Module] = list(
            modules
            or [
                SenseModule(),
                InteroModule(),
                AffectModule(),
                WorldModelModule(),
                SimulationModule(),
                MemoryBridgeModule(),
                KnowledgeBridgeModule(),
                AttentionModule(),
                WorkspaceCompetitionModule(),
                WorkingSetModule(),
                PolicyModule(),
                SelfModelExtModule(),
                MetaModule(),
                ReportModule(),
                PhenomenologyProbeModule(),
                AutotuneModule(),
            ]
        )

    def set_runtime_overrides(self, overrides: Mapping[str, Any] | None) -> None:
        self._runtime_overrides = dict(overrides or {})
        self._refresh_config()

    def _refresh_config(self) -> None:
        if self._respect_tuned_overlay:
            tuned_overlay, invalid = load_tuned_overlay(self.state_store)
            self._tuned_overlay = dict(tuned_overlay)
            invalid_tuple = tuple(sorted(str(k) for k in invalid))
            if invalid_tuple and invalid_tuple != self._last_invalid_overlay_keys:
                bus.append(
                    self.state_dir,
                    "consciousness.param_invalid",
                    {"invalid_keys": list(invalid_tuple)},
                    tags=["consciousness", "config"],
                )
            self._last_invalid_overlay_keys = invalid_tuple
        else:
            self._tuned_overlay = {}
        self.config = resolve_config(
            self._base_config,
            tuned_overlay=self._tuned_overlay,
            runtime_overrides=self._runtime_overrides,
        )

    def register_perturbation(self, payload: Mapping[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        ts = str(payload.get("ts") or now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        duration = max(0.0, float(payload.get("duration_s") or 0.0))
        expires_at = now
        if duration > 0.0:
            expires_at = now + timedelta(seconds=duration)
        self._active_perturbations.append(
            {
                "id": str(payload.get("id") or uuid.uuid4().hex),
                "kind": str(payload.get("kind") or ""),
                "target": str(payload.get("target") or "*"),
                "magnitude": float(payload.get("magnitude") or 0.0),
                "duration_s": duration,
                "meta": dict(payload.get("meta") or {}),
                "ts": ts,
                "_expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    def _refresh_perturbations(self, now: datetime) -> None:
        kept: list[dict[str, Any]] = []
        for row in self._active_perturbations:
            exp_text = str(row.get("_expires_at") or "")
            expires_at: datetime | None = None
            if exp_text:
                try:
                    if exp_text.endswith("Z"):
                        exp_text = exp_text[:-1] + "+00:00"
                    expires_at = datetime.fromisoformat(exp_text)
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    else:
                        expires_at = expires_at.astimezone(timezone.utc)
                except Exception:
                    expires_at = None
            if expires_at is not None and now > expires_at:
                continue
            kept.append(row)
        self._active_perturbations = kept

    def _module_period(self, module_name: str) -> int:
        periods = self.config.get("module_tick_periods")
        if isinstance(periods, Mapping):
            value = periods.get(module_name)
            if isinstance(value, (int, float)):
                return max(1, int(value))
        direct = self.config.get(f"module_period_{module_name}")
        if isinstance(direct, (int, float)):
            return max(1, int(direct))
        return 1

    def _module_disabled(self, module_name: str) -> bool:
        raw = self.config.get("disable_modules")
        if isinstance(raw, (list, tuple, set)):
            disabled = {str(x) for x in raw}
            return module_name in disabled
        return False

    def _collect_context(self) -> TickContext:
        self._refresh_config()
        event_limit = int(self.config.get("recent_events_limit", 300))
        broadcast_limit = int(self.config.get("recent_broadcast_limit", 300))
        events = bus.iter_events(self.state_dir, limit=event_limit)
        broadcasts = workspace.iter_broadcast(self.state_dir, limit=broadcast_limit)
        return TickContext(
            state_dir=self.state_dir,
            config=self.config,
            recent_events=events,
            recent_broadcasts=broadcasts,
            rng=self.rng,
            beat_count=self.beat_count,
            state_store=self.state_store,
            active_perturbations=list(self._active_perturbations),
            now=datetime.now(timezone.utc),
        )

    def tick(self) -> KernelResult:
        now = datetime.now(timezone.utc)
        self._refresh_perturbations(now)
        ctx = self._collect_context()
        errors: List[str] = []
        names: List[str] = []
        for module in self.modules:
            if self._module_disabled(module.name):
                continue
            period = self._module_period(module.name)
            if period > 1 and (self.beat_count % period) != 0:
                continue
            names.append(module.name)
            try:
                module.tick(ctx)
            except Exception as exc:  # pragma: no cover - defensive safety path
                msg = f"{module.name}: {exc}"
                errors.append(msg)
                ctx.emit_event(
                    "consciousness.module_error",
                    {"module": module.name, "error": str(exc)},
                    tags=["consciousness", "module_error"],
                )

        self.beat_count += 1
        self.state_store.set_meta("beat_count", self.beat_count)
        self.state_store.flush()

        return KernelResult(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            emitted_events=len(ctx.emitted_events),
            modules=names,
            errors=errors,
        )
