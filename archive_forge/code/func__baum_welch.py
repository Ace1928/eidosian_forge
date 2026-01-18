import warnings
from Bio import BiopythonDeprecationWarning
def _baum_welch(N, M, training_outputs, p_initial=None, p_transition=None, p_emission=None, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None, update_fn=None):
    """Implement the Baum-Welch algorithm to evaluate unknown parameters in the MarkovModel object (PRIVATE)."""
    if p_initial is None:
        p_initial = _random_norm(N)
    else:
        p_initial = _copy_and_check(p_initial, (N,))
    if p_transition is None:
        p_transition = _random_norm((N, N))
    else:
        p_transition = _copy_and_check(p_transition, (N, N))
    if p_emission is None:
        p_emission = _random_norm((N, M))
    else:
        p_emission = _copy_and_check(p_emission, (N, M))
    lp_initial = np.log(p_initial)
    lp_transition = np.log(p_transition)
    lp_emission = np.log(p_emission)
    if pseudo_initial is not None:
        lpseudo_initial = np.log(pseudo_initial)
    else:
        lpseudo_initial = None
    if pseudo_transition is not None:
        lpseudo_transition = np.log(pseudo_transition)
    else:
        lpseudo_transition = None
    if pseudo_emission is not None:
        lpseudo_emission = np.log(pseudo_emission)
    else:
        lpseudo_emission = None
    prev_llik = None
    for i in range(MAX_ITERATIONS):
        llik = LOG0
        for outputs in training_outputs:
            llik += _baum_welch_one(N, M, outputs, lp_initial, lp_transition, lp_emission, lpseudo_initial, lpseudo_transition, lpseudo_emission)
        if update_fn is not None:
            update_fn(i, llik)
        if prev_llik is not None and np.fabs(prev_llik - llik) < 0.1:
            break
        prev_llik = llik
    else:
        raise RuntimeError('HMM did not converge in %d iterations' % MAX_ITERATIONS)
    return [np.exp(_) for _ in (lp_initial, lp_transition, lp_emission)]