from __future__ import annotations
from typing import NoReturn, TypeVar
from attrs import define as _define, frozen as _frozen
class UnsupportedSubclassing(Exception):

    def __str__(self):
        return "Subclassing is not part of referencing's public API. If no other suitable API exists for what you're trying to do, feel free to file an issue asking for one."