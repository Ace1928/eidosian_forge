from __future__ import annotations

from ..types import TickContext


class WorldModelModule:
    name = "world_model"

    def tick(self, ctx: TickContext) -> None:
        return None
