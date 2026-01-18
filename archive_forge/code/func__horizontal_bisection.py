from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _horizontal_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Horizontal bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = (a, b)
    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q
    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2
    y = (v + t) / 2
    f1H = dmp_eval_in(f1, y, 1, 1, F)
    f2H = dmp_eval_in(f2, y, 1, 1, F)
    I_H = dup_isolate_real_roots_list([f1H, f2H], F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L1_B, I_L1_U = (I_L1, I_H)
    I_L2_B, I_L2_U = ([], [])
    I_L3_B, I_L3_U = (_reverse_intervals(I_H), I_L3)
    I_L4_B, I_L4_U = ([], [])
    for I in I_L2:
        (a, b), indices, h = I
        if a == b:
            if a == y:
                I_L2_B.append(I)
                I_L2_U.append(I)
            elif a < y:
                I_L2_B.append(I)
            else:
                I_L2_U.append(I)
        elif b <= y:
            I_L2_B.append(I)
        elif a >= y:
            I_L2_U.append(I)
        else:
            a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)
            if b <= y:
                I_L2_B.append(((a, b), indices, h))
            if a >= y:
                I_L2_U.append(((a, b), indices, h))
    for I in I_L4:
        (b, a), indices, h = I
        if a == b:
            if a == y:
                I_L4_B.append(I)
                I_L4_U.append(I)
            elif a < y:
                I_L4_B.append(I)
            else:
                I_L4_U.append(I)
        elif b <= y:
            I_L4_B.append(I)
        elif a >= y:
            I_L4_U.append(I)
        else:
            a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)
            if b <= y:
                I_L4_B.append(((b, a), indices, h))
            if a >= y:
                I_L4_U.append(((b, a), indices, h))
    Q_L1_B = Q_L1
    Q_L2_B = _intervals_to_quadrants(I_L2_B, f1L2F, f2L2F, v, y, F)
    Q_L3_B = _intervals_to_quadrants(I_L3_B, f1H, f2H, s, u, F)
    Q_L4_B = _intervals_to_quadrants(I_L4_B, f1L4F, f2L4F, y, v, F)
    Q_L1_U = _intervals_to_quadrants(I_L1_U, f1H, f2H, u, s, F)
    Q_L2_U = _intervals_to_quadrants(I_L2_U, f1L2F, f2L2F, y, t, F)
    Q_L3_U = Q_L3
    Q_L4_U = _intervals_to_quadrants(I_L4_U, f1L4F, f2L4F, t, y, F)
    T_B = _traverse_quadrants(Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B, exclude=True)
    T_U = _traverse_quadrants(Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U, exclude=True)
    N_B = _winding_number(T_B, F)
    N_U = _winding_number(T_U, F)
    I_B = (I_L1_B, I_L2_B, I_L3_B, I_L4_B)
    Q_B = (Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B)
    I_U = (I_L1_U, I_L2_U, I_L3_U, I_L4_U)
    Q_U = (Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U)
    F1_B = (f1L1F, f1L2F, f1H, f1L4F)
    F2_B = (f2L1F, f2L2F, f2H, f2L4F)
    F1_U = (f1H, f1L2F, f1L3F, f1L4F)
    F2_U = (f2H, f2L2F, f2L3F, f2L4F)
    a, b = ((u, v), (s, y))
    c, d = ((u, y), (s, t))
    D_B = (N_B, a, b, I_B, Q_B, F1_B, F2_B)
    D_U = (N_U, c, d, I_U, Q_U, F1_U, F2_U)
    return (D_B, D_U)