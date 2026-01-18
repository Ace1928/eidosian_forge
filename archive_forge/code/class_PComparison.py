from __future__ import annotations
import typing
class PComparison(Protocol):
    """
        Objects that can be compaired
        """

    def __eq__(self, other, /) -> bool:
        ...

    def __lt__(self, other, /) -> bool:
        ...

    def __gt__(self, other, /) -> bool:
        ...