from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _iis_solve_delta(N, feature, f_sharp, empirical, prob_yx, max_newton_iterations, newton_converge):
    """Solve delta using Newton's method (PRIVATE)."""
    delta = 0.0
    iters = 0
    while iters < max_newton_iterations:
        f_newton = df_newton = 0.0
        for (i, j), f in feature.items():
            prod = prob_yx[i][j] * f * np.exp(delta * f_sharp[i][j])
            f_newton += prod
            df_newton += prod * f_sharp[i][j]
        f_newton, df_newton = (empirical - f_newton / N, -df_newton / N)
        ratio = f_newton / df_newton
        delta -= ratio
        if np.fabs(ratio) < newton_converge:
            break
        iters = iters + 1
    else:
        raise RuntimeError("Newton's method did not converge")
    return delta