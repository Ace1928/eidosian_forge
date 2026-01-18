import fractions
import pytest
import cirq
class ModifiedHash(tuple):

    def __hash__(self):
        return super().__hash__() ^ 1