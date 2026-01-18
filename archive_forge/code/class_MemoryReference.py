from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class MemoryReference(QuilAtom, Expression):
    """
    Representation of a reference to a classical memory address.

    :param name: The name of the variable
    :param offset: Everything in Quil is a C-style array, so every memory reference has an offset.
    :param declared_size: The optional size of the named declaration. This can be used for bounds
        checking, but isn't. It is used for pretty-printing to quil by deciding whether to output
        memory references with offset 0 as either e.g. ``ro[0]`` or ``beta`` depending on whether
        the declared variable is of length >1 or 1, resp.
    """

    def __init__(self, name: str, offset: int=0, declared_size: Optional[int]=None):
        if not isinstance(offset, int) or offset < 0:
            raise TypeError('MemoryReference offset must be a non-negative int')
        self.name = name
        self.offset = offset
        self.declared_size = declared_size

    def out(self) -> str:
        if self.declared_size is not None and self.declared_size == 1 and (self.offset == 0):
            return '{}'.format(self.name)
        else:
            return '{}[{}]'.format(self.name, self.offset)

    def __str__(self) -> str:
        if self.declared_size is not None and self.declared_size == 1 and (self.offset == 0):
            return '{}'.format(self.name)
        else:
            return '{}[{}]'.format(self.name, self.offset)

    def __repr__(self) -> str:
        return '<MRef {}[{}]>'.format(self.name, self.offset)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryReference) and other.name == self.name and (other.offset == self.offset)

    def __hash__(self) -> int:
        return hash((self.name, self.offset))

    def __getitem__(self, offset: int) -> 'MemoryReference':
        if self.offset != 0:
            raise ValueError('Please only index off of the base MemoryReference (offset = 0)')
        if self.declared_size and offset >= self.declared_size:
            raise IndexError('MemoryReference index out of range')
        return MemoryReference(name=self.name, offset=offset)

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union['MemoryReference', ExpressionValueDesignator]:
        if self in d:
            return d[self]
        return self