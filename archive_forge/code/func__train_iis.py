from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _train_iis(xs, classes, features, f_sharp, alphas, e_empirical, max_newton_iterations, newton_converge):
    """Do one iteration of hill climbing to find better alphas (PRIVATE)."""
    p_yx = _calc_p_class_given_x(xs, classes, features, alphas)
    N = len(xs)
    newalphas = alphas[:]
    for i in range(len(alphas)):
        delta = _iis_solve_delta(N, features[i], f_sharp, e_empirical[i], p_yx, max_newton_iterations, newton_converge)
        newalphas[i] += delta
    return newalphas