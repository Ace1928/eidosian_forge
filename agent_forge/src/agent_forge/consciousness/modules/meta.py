from __future__ import annotations

from ..types import TickContext


class MetaModule:
    name = "meta"

    def tick(self, ctx: TickContext) -> None:
        return None
