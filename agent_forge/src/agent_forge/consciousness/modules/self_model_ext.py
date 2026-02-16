from __future__ import annotations

from ..types import TickContext


class SelfModelExtModule:
    name = "self_model_ext"

    def tick(self, ctx: TickContext) -> None:
        return None
