from __future__ import annotations
import os
import sys
from typing import Protocol, Sequence, TypeVar, Union
class SupportsRead(Protocol[_T_co]):

    def read(self, __length: int=...) -> _T_co:
        ...