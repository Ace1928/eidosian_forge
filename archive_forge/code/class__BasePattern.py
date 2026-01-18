from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class _BasePattern(ABC):
    __slots__ = ('_alphabet_cache', '_prefix_cache', '_lengths_cache')

    @abstractmethod
    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=None) -> FSM:
        raise NotImplementedError

    @abstractmethod
    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        raise NotImplementedError

    def get_alphabet(self, flags: REFlags) -> Alphabet:
        if not hasattr(self, '_alphabet_cache'):
            super(_BasePattern, self).__setattr__('_alphabet_cache', {})
        if flags not in self._alphabet_cache:
            self._alphabet_cache[flags] = self._get_alphabet(flags)
        return self._alphabet_cache[flags]

    @abstractmethod
    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        raise NotImplementedError

    @property
    def prefix_postfix(self) -> Tuple[int, Optional[int]]:
        """Returns the number of dots that have to be pre-/postfixed to support look(aheads|backs)"""
        if not hasattr(self, '_prefix_cache'):
            super(_BasePattern, self).__setattr__('_prefix_cache', self._get_prefix_postfix())
        return self._prefix_cache

    @abstractmethod
    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        raise NotImplementedError

    @property
    def lengths(self) -> Tuple[int, Optional[int]]:
        """Returns the minimum and maximum length that this pattern can match
         (maximum can be None bei infinite length)"""
        if not hasattr(self, '_lengths_cache'):
            super(_BasePattern, self).__setattr__('_lengths_cache', self._get_lengths())
        return self._lengths_cache

    @abstractmethod
    def simplify(self) -> '_BasePattern':
        raise NotImplementedError