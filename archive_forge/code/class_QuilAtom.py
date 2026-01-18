from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class QuilAtom(object):
    """
    Abstract class for atomic elements of Quil.
    """

    def out(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        raise NotImplementedError()