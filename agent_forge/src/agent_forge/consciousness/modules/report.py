from __future__ import annotations

from ..types import TickContext


class ReportModule:
    name = "report"

    def tick(self, ctx: TickContext) -> None:
        return None
