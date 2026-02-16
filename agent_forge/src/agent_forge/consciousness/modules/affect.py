from __future__ import annotations

from ..types import TickContext


class AffectModule:
    name = "affect"

    def tick(self, ctx: TickContext) -> None:
        return None
