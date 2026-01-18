from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _as_pair(atom):
    if isinstance(atom, Not):
        return (atom.arg, False)
    else:
        return (atom, True)