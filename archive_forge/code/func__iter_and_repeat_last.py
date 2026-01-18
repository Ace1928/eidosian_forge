from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
def _iter_and_repeat_last(one_iter: Iterator[Params]):
    last = None
    for last in one_iter:
        yield last
    while True:
        yield last