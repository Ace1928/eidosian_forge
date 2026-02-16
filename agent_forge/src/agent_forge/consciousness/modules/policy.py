from __future__ import annotations

from ..types import TickContext


class PolicyModule:
    name = "policy"

    def tick(self, ctx: TickContext) -> None:
        return None
