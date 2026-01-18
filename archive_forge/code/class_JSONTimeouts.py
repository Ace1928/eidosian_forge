from typing import TYPE_CHECKING
class JSONTimeouts(TypedDict, total=False):
    implicit: int
    pageLoad: int
    script: int