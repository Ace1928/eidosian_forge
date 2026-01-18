from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def _overlaps(r1, r2):
    len1 = len(r1)
    len2 = len(r2)
    result = []
    for j in range(1, len1 + len2):
        if r1.subword(len1 - j, len1 + len2 - j, strict=False) == r2.subword(j - len1, j, strict=False):
            a = r1.subword(0, len1 - j, strict=False)
            a = a * r2.subword(0, j - len1, strict=False)
            b = r2.subword(j - len1, j, strict=False)
            c = r2.subword(j, len2, strict=False)
            c = c * r1.subword(len1 + len2 - j, len1, strict=False)
            result.append(a * b * c)
    return result