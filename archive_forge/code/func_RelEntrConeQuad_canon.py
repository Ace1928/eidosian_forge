from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
def RelEntrConeQuad_canon(con: RelEntrConeQuad, args) -> Tuple[Constraint, List[Constraint]]:
    """
    Use linear and SOC constraints to approximately enforce
        con.x * log(con.x / con.y) <= con.z.

    We rely on an SOC characterization of 2-by-2 PSD matrices.
    Namely, a matrix
        [ a, b ]
        [ b, c ]
    is PSD if and only if (a, c) >= 0 and a*c >= b**2.
    That system of constraints can be expressed as
        a >= quad_over_lin(b, c).

    Note: constraint canonicalization in CVXPY uses a return format
    (lead_con, con_list) where lead_con is a Constraint that might be
    used in dual variable recovery and con_list consists of extra
    Constraint objects as needed.
    """
    k, m = (con.k, con.m)
    x, y = (con.x, con.y)
    n = x.size
    Z = Variable(shape=(k + 1, n))
    w, t = gauss_legendre(m)
    T = Variable(shape=(m, n))
    lead_con = Zero(w @ T + con.z / 2 ** k)
    constrs = [Zero(Z[0] - y)]
    for i in range(k):
        epi = Z[i, :]
        stackedZ = Z[i + 1, :]
        cons = rotated_quad_cone(stackedZ, epi, x)
        constrs.append(cons)
        constrs.extend([epi >= 0, x >= 0])
    for i in range(m):
        off_diag = -t[i] ** 0.5 * T[i, :]
        epi = Z[k, :] - x - T[i, :]
        cons = rotated_quad_cone(off_diag, epi, x - t[i] * T[i, :])
        constrs.append(cons)
        constrs.extend([epi >= 0, x - t[i] * T[i, :] >= 0])
    return (lead_con, constrs)