import numpy as np
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.solution import Solution
class Exotic2Common(Canonicalization):
    CANON_METHODS = {PowConeND: pow_nd_canon}

    def __init__(self, problem=None) -> None:
        super(Exotic2Common, self).__init__(problem=problem, canon_methods=Exotic2Common.CANON_METHODS)

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid] for orig_id, vid in inverse_data.cons_id_map.items() if vid in solution.dual_vars}
        if dvars == {}:
            return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)
        dv = {}
        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND):
                div_size = int(dvars[cons_id].shape[1] // cons.args[1].shape[0])
                dv[cons_id] = []
                for i in range(cons.args[1].shape[0]):
                    dv[cons_id].append([])
                    tmp_duals = dvars[cons_id][:, i * div_size:(i + 1) * div_size]
                    for j, col_dvars in enumerate(tmp_duals.T):
                        if j == len(tmp_duals.T) - 1:
                            dv[cons_id][-1] += [col_dvars[0], col_dvars[1]]
                        else:
                            dv[cons_id][-1].append(col_dvars[0])
                    dv[cons_id][-1].append(tmp_duals.T[0][-1])
                dvars[cons_id] = np.array(dv[cons_id])
        return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)