import fractions
import numpy as np
import pytest
import sympy
import cirq
class SubsAwareSymbol(sympy.Symbol):
    """A Symbol that registers a call to its `subs` method."""

    def __init__(self, sym: str):
        self.called = False
        self.symbol = sympy.Symbol(sym)

    def subs(self, *args, **kwargs):
        self.called = True
        return self.symbol.subs(*args, **kwargs)