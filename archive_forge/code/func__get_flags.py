from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def _get_flags(plus: str) -> REFlags:
    res = REFlags(0)
    for c in plus:
        try:
            res |= _flags[c]
        except KeyError:
            raise Unsupported(f'Flag {c} is not implemented')
    return res