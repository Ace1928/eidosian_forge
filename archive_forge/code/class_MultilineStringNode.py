from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
@dataclass(unsafe_hash=True)
class MultilineStringNode(BaseStringNode):

    def __init__(self, token: Token[str]):
        super().__init__(token)
        self.value = token.value