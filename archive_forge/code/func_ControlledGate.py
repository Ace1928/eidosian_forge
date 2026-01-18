from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def ControlledGate(ctrls, target):
    return CGate(tuple(ctrls), onequbitgate(target))