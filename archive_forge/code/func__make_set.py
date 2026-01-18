from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
def _make_set(ops):
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops