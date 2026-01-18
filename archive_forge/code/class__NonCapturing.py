from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class _NonCapturing:
    """Represents a lookahead/lookback. Matches `inner` without 'consuming' anything. Can be negated.
    Only valid inside a `_Concatenation`"""
    inner: _BasePattern
    backwards: bool
    negate: bool
    __slots__ = ('inner', 'backwards', 'negate')

    def get_alphabet(self, flags: REFlags) -> Alphabet:
        return self.inner.get_alphabet(flags)

    def simplify(self) -> '_NonCapturing':
        return self.__class__(self.inner.simplify(), self.backwards, self.negate)