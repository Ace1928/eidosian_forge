from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
def _check_duplicate_keys(sweeps):
    keys = set()
    for sweep in sweeps:
        if any((key in keys for key in sweep.keys)):
            raise ValueError('duplicate keys')
        keys.update(sweep.keys)