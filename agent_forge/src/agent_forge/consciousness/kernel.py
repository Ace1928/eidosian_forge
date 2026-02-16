from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agent_forge.core import events as bus
from agent_forge.core import workspace

from .modules.attention import AttentionModule
from .modules.meta import MetaModule
from .modules.policy import PolicyModule
from .modules.report import ReportModule
from .modules.self_model_ext import SelfModelExtModule
from .modules.world_model import WorldModelModule
from .modules.workspace_competition import WorkspaceCompetitionModule
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
    ) -> None:
        self.state_dir = Path(state_dir)
        self.config = merged_config(config or {})
        self.rng = random.Random(seed)
        self.modules: List[Module] = list(
            modules
            or [
                WorldModelModule(),
                AttentionModule(),
                WorkspaceCompetitionModule(),
                PolicyModule(),
                SelfModelExtModule(),
                MetaModule(),
                ReportModule(),
            ]
        )

    def _collect_context(self) -> TickContext:
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
            now=datetime.now(timezone.utc),
        )

    def tick(self) -> KernelResult:
        ctx = self._collect_context()
        errors: List[str] = []
        names: List[str] = []
        for module in self.modules:
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

        return KernelResult(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            emitted_events=len(ctx.emitted_events),
            modules=names,
            errors=errors,
        )
