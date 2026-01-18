from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
class CreateOneQubitGate(type):

    def __new__(mcl, name, latexname=None):
        if not latexname:
            latexname = name
        return type(name + 'Gate', (OneQubitGate,), {'gate_name': name, 'gate_name_latex': latexname})