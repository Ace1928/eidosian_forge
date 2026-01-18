from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
class PbxComment:

    def __init__(self, text: str):
        assert isinstance(text, str)
        assert '/*' not in text
        self.text = f'/* {text} */'

    def write(self, ofile: T.TextIO, indent_level: int) -> None:
        ofile.write(f'\n{self.text}\n')