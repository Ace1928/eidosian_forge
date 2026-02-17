from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from agent_forge.core import events as bus
from agent_forge.core import workspace

from .modules.attention import AttentionModule
from .modules.affect import AffectModule
from .modules.autotune import AutotuneModule
from .modules.experiment_designer import ExperimentDesignerModule
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
                ExperimentDesignerModule(),
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

    def _watchdog_enabled(self) -> bool:
        return bool(self.config.get("kernel_watchdog_enabled", True))

    def _watchdog_modules_state(self) -> MutableMapping[str, Any]:
        ns = self.state_store.namespace("__kernel_watchdog__", defaults={"modules": {}})
        modules = ns.get("modules")
        if not isinstance(modules, MutableMapping):
            modules = {}
            ns["modules"] = modules
            self.state_store.mark_dirty()
        return modules

    def _watchdog_module_state(self, module_name: str) -> MutableMapping[str, Any]:
        modules = self._watchdog_modules_state()
        module_key = str(module_name)
        state = modules.get(module_key)
        if not isinstance(state, MutableMapping):
            state = {
                "consecutive_errors": 0,
                "total_errors": 0,
                "quarantine_count": 0,
                "recoveries": 0,
                "quarantined_until_beat": 0,
                "last_error": "",
                "last_error_ts": "",
            }
            modules[module_key] = state
            self.state_store.mark_dirty()
        return state

    def _handle_watchdog_release(self, module_name: str, ctx: TickContext) -> bool:
        if not self._watchdog_enabled():
            return False
        state = self._watchdog_module_state(module_name)
        quarantined_until = int(state.get("quarantined_until_beat") or 0)
        if quarantined_until <= 0:
            return False
        if self.beat_count < quarantined_until:
            return True

        state["quarantined_until_beat"] = 0
        state["recoveries"] = int(state.get("recoveries") or 0) + 1
        state["consecutive_errors"] = 0
        self.state_store.mark_dirty()
        ctx.emit_event(
            "consciousness.module_recovered",
            {
                "module": module_name,
                "beat": int(self.beat_count),
                "recoveries": int(state.get("recoveries") or 0),
            },
            tags=["consciousness", "watchdog", "recovery"],
        )
        return False

    def _register_module_success(self, module_name: str) -> None:
        if not self._watchdog_enabled():
            return
        state = self._watchdog_module_state(module_name)
        if int(state.get("consecutive_errors") or 0) != 0:
            state["consecutive_errors"] = 0
            self.state_store.mark_dirty()

    def _register_module_error(self, module_name: str, error_text: str, ctx: TickContext) -> None:
        state = self._watchdog_module_state(module_name)
        consecutive_errors = int(state.get("consecutive_errors") or 0) + 1
        total_errors = int(state.get("total_errors") or 0) + 1
        state["consecutive_errors"] = consecutive_errors
        state["total_errors"] = total_errors
        state["last_error"] = str(error_text)
        state["last_error_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.state_store.mark_dirty()

        quarantined_until = int(state.get("quarantined_until_beat") or 0)
        threshold = max(
            1,
            int(self.config.get("kernel_watchdog_max_consecutive_errors", 3) or 3),
        )
        quarantine_beats = max(
            1,
            int(self.config.get("kernel_watchdog_quarantine_beats", 20) or 20),
        )
        quarantined = False
        if self._watchdog_enabled() and consecutive_errors >= threshold:
            next_until = int(self.beat_count) + quarantine_beats + 1
            if next_until > quarantined_until:
                state["quarantined_until_beat"] = next_until
                state["quarantine_count"] = int(state.get("quarantine_count") or 0) + 1
                quarantined_until = next_until
                quarantined = True
                self.state_store.mark_dirty()

        ctx.emit_event(
            "consciousness.module_error",
            {
                "module": module_name,
                "error": str(error_text),
                "consecutive_errors": int(consecutive_errors),
                "total_errors": int(total_errors),
                "quarantined_until_beat": int(quarantined_until),
            },
            tags=["consciousness", "module_error"],
        )
        if quarantined:
            ctx.emit_event(
                "consciousness.module_quarantined",
                {
                    "module": module_name,
                    "beat": int(self.beat_count),
                    "consecutive_errors": int(consecutive_errors),
                    "threshold": int(threshold),
                    "quarantined_until_beat": int(quarantined_until),
                    "quarantine_beats": int(quarantine_beats),
                },
                tags=["consciousness", "watchdog", "quarantine"],
            )

    @staticmethod
    def _watchdog_status_from_snapshot(
        *,
        snapshot: Mapping[str, Any],
        config: Mapping[str, Any],
        beat_count: int,
    ) -> dict[str, Any]:
        modules_root = snapshot.get("modules")
        watchdog_root: Mapping[str, Any] = {}
        if isinstance(modules_root, Mapping):
            raw_watchdog = modules_root.get("__kernel_watchdog__")
            if isinstance(raw_watchdog, Mapping):
                watchdog_root = raw_watchdog
        module_rows_raw = watchdog_root.get("modules")
        rows: list[dict[str, Any]] = []
        if isinstance(module_rows_raw, Mapping):
            for module_name, row in module_rows_raw.items():
                if not isinstance(row, Mapping):
                    continue
                quarantined_until = int(row.get("quarantined_until_beat") or 0)
                rows.append(
                    {
                        "module": str(module_name),
                        "consecutive_errors": int(row.get("consecutive_errors") or 0),
                        "total_errors": int(row.get("total_errors") or 0),
                        "quarantine_count": int(row.get("quarantine_count") or 0),
                        "recoveries": int(row.get("recoveries") or 0),
                        "quarantined_until_beat": quarantined_until,
                        "quarantined": quarantined_until > int(beat_count),
                        "last_error": str(row.get("last_error") or ""),
                        "last_error_ts": str(row.get("last_error_ts") or ""),
                    }
                )
        rows.sort(key=lambda item: item["module"])
        quarantined = [row for row in rows if bool(row.get("quarantined"))]
        return {
            "enabled": bool(config.get("kernel_watchdog_enabled", True)),
            "max_consecutive_errors": int(
                config.get("kernel_watchdog_max_consecutive_errors", 3) or 3
            ),
            "quarantine_beats": int(
                config.get("kernel_watchdog_quarantine_beats", 20) or 20
            ),
            "beat_count": int(beat_count),
            "module_count": len(rows),
            "quarantined_modules": len(quarantined),
            "modules": rows,
            "quarantined_module_names": [str(row.get("module") or "") for row in quarantined],
            "total_errors": sum(int(row.get("total_errors") or 0) for row in rows),
        }

    @staticmethod
    def _payload_safety_status_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "max_payload_bytes": int(
                config.get("consciousness_max_payload_bytes", 16384) or 16384
            ),
            "max_depth": int(config.get("consciousness_max_depth", 8) or 8),
            "max_collection_items": int(
                config.get("consciousness_max_collection_items", 64) or 64
            ),
            "max_string_chars": int(
                config.get("consciousness_max_string_chars", 4096) or 4096
            ),
            "truncation_event_enabled": bool(
                config.get("consciousness_payload_truncation_event", True)
            ),
        }

    @classmethod
    def read_runtime_health(
        cls,
        state_dir: str | Path,
        *,
        config: Optional[Mapping[str, Any]] = None,
        respect_tuned_overlay: bool = True,
    ) -> dict[str, Any]:
        state_path = Path(state_dir)
        base_config = merged_config(config or {})
        store = ModuleStateStore(
            state_path,
            autosave_interval_secs=float(
                base_config.get("state_autosave_interval_secs", 2.0)
            ),
        )
        tuned_overlay: Mapping[str, Any] = {}
        if respect_tuned_overlay:
            loaded_overlay, _ = load_tuned_overlay(store)
            tuned_overlay = loaded_overlay
        resolved_config = resolve_config(
            base_config,
            tuned_overlay=tuned_overlay,
            runtime_overrides={},
        )
        beat_count = int(store.get_meta("beat_count", 0) or 0)
        snapshot = store.snapshot()
        return {
            "beat_count": int(beat_count),
            "watchdog": cls._watchdog_status_from_snapshot(
                snapshot=snapshot,
                config=resolved_config,
                beat_count=beat_count,
            ),
            "payload_safety": cls._payload_safety_status_from_config(
                resolved_config
            ),
        }

    def watchdog_status(self) -> dict[str, Any]:
        return self._watchdog_status_from_snapshot(
            snapshot=self.state_store.snapshot(),
            config=self.config,
            beat_count=int(self.beat_count),
        )

    def payload_safety_status(self) -> dict[str, Any]:
        return self._payload_safety_status_from_config(self.config)

    def runtime_health(self) -> dict[str, Any]:
        return {
            "beat_count": int(self.beat_count),
            "watchdog": self.watchdog_status(),
            "payload_safety": self.payload_safety_status(),
        }

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
            if self._handle_watchdog_release(module.name, ctx):
                continue
            period = self._module_period(module.name)
            if period > 1 and (self.beat_count % period) != 0:
                continue
            names.append(module.name)
            try:
                module.tick(ctx)
                self._register_module_success(module.name)
            except Exception as exc:  # pragma: no cover - defensive safety path
                msg = f"{module.name}: {exc}"
                errors.append(msg)
                self._register_module_error(module.name, str(exc), ctx)

        self.beat_count += 1
        self.state_store.set_meta("beat_count", self.beat_count)
        self.state_store.flush()

        return KernelResult(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            emitted_events=len(ctx.emitted_events),
            modules=names,
            errors=errors,
        )
