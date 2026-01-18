import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def _normalize1(sign, man, exp, bc, prec, rnd):
    """same as normalize, but with the added condition that
       man is odd or zero
    """
    if not man:
        return fzero
    if bc <= prec:
        return (sign, man, exp, bc)
    n = bc - prec
    if rnd == round_nearest:
        t = man >> n - 1
        if t & 1 and (t & 2 or man & h_mask[n < 300][n]):
            man = (t >> 1) + 1
        else:
            man = t >> 1
    elif shifts_down[rnd][sign]:
        man >>= n
    else:
        man = -(-man >> n)
    exp += n
    bc = prec
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    if man == 1:
        bc = 1
    return (sign, man, exp, bc)