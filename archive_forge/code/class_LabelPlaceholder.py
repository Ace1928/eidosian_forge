from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class LabelPlaceholder(QuilAtom):

    def __init__(self, prefix: str='L'):
        self.prefix = prefix

    def out(self) -> str:
        raise RuntimeError('Label has not been assigned a name')

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return '<LabelPlaceholder {} {}>'.format(self.prefix, id(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelPlaceholder) and id(other) == id(self)

    def __hash__(self) -> int:
        return hash(id(self))