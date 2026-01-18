from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _perform_word_moves(matrices, G):
    mats = [None] + matrices
    moves = G._word_moves()
    while moves:
        a = moves.pop(0)
        if a >= len(mats):
            n = moves.index(a)
            word, moves = (moves[:n], moves[n + 1:])
            mats.append(prod([mats[g] if g > 0 else _adjoint2(mats[-g]) for g in word]))
        else:
            b = moves.pop(0)
            if a == b:
                mats[a] = mats[-1]
                mats = mats[:-1]
            elif a == -b:
                mats[a] = _adjoint2(mats[a])
            else:
                A, B = (mats[abs(a)], mats[abs(b)])
                if a * b < 0:
                    B = _adjoint2(B)
                mats[abs(a)] = A * B if a > 0 else B * A
    return mats[1:G.num_generators() + 1]