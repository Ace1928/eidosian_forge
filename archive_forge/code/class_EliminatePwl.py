from cvxpy.atoms import abs, max, maximum, norm1, norm_inf, sum_largest
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.eliminate_pwl.canonicalizers import CANON_METHODS as elim_pwl_methods
class EliminatePwl(Canonicalization):
    """Eliminates piecewise linear atoms."""

    def __init__(self, problem=None) -> None:
        super(EliminatePwl, self).__init__(problem=problem, canon_methods=elim_pwl_methods)

    def accepts(self, problem) -> bool:
        atom_types = [type(atom) for atom in problem.atoms()]
        pwl_types = [abs, maximum, sum_largest, max, norm1, norm_inf]
        return any((atom in pwl_types for atom in atom_types))

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError('Cannot canonicalize pwl atoms.')
        return super(EliminatePwl, self).apply(problem)