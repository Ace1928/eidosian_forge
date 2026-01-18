from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class QubitPlaceholder(QuilAtom):

    def out(self) -> str:
        raise RuntimeError('Qubit {} has not been assigned an index'.format(self))

    @property
    def index(self) -> NoReturn:
        raise RuntimeError('Qubit {} has not been assigned an index'.format(self))

    def __str__(self) -> str:
        return 'q{}'.format(id(self))

    def __repr__(self) -> str:
        return '<QubitPlaceholder {}>'.format(id(self))

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QubitPlaceholder) and id(other) == id(self)

    @classmethod
    def register(cls, n: int) -> List['QubitPlaceholder']:
        """Return a 'register' of ``n`` QubitPlaceholders.

        >>> qs = QubitPlaceholder.register(8) # a qubyte
        >>> prog = Program(H(q) for q in qs)
        >>> address_qubits(prog).out()
        H 0
        H 1
        ...
        >>>

        The returned register is a Python list of QubitPlaceholder objects, so all
        normal list semantics apply.

        :param n: The number of qubits in the register
        """
        return [cls() for _ in range(n)]