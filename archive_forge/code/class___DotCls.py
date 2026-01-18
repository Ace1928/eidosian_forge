from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class __DotCls(_Repeatable):

    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=REFlags(0)) -> FSM:
        if alphabet is None:
            alphabet = self.get_alphabet(flags)
        if flags is None or not flags & REFlags.SINGLE_LINE:
            symbols = set(alphabet) - {'\n'}
        else:
            symbols = alphabet
        return FSM(alphabet=alphabet, states={0, 1}, initial=0, finals={1}, map={0: {alphabet[sym]: 1 for sym in symbols}})

    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        if flags & REFlags.SINGLE_LINE:
            return Alphabet.from_groups({anything_else})
        else:
            return Alphabet.from_groups({anything_else}, {'\n'})

    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        return (0, 0)

    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        return (1, 1)

    def simplify(self) -> '__DotCls':
        return self