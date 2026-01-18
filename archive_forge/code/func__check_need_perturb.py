from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
def _check_need_perturb(ctx, terms, prec, discard_known_zeros):
    perturb = recompute = False
    extraprec = 0
    discard = []
    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        have_singular_nongamma_weight = False
        for k, w in enumerate(w_s):
            if not w:
                if ctx.re(c_s[k]) <= 0 and c_s[k]:
                    perturb = recompute = True
                    have_singular_nongamma_weight = True
        pole_count = [0, 0, 0]
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = ctx.nint_distance(x)
                if n > 0:
                    continue
                if d == ctx.ninf:
                    ok = False
                    if data_index == 2:
                        for u in a_s:
                            if ctx.isnpint(u) and u >= int(n):
                                ok = True
                                break
                    if ok:
                        continue
                    pole_count[data_index] += 1
                elif d < -4:
                    extraprec += -d
                    recompute = True
        if discard_known_zeros and pole_count[1] > pole_count[0] + pole_count[2] and (not have_singular_nongamma_weight):
            discard.append(term_index)
        elif sum(pole_count):
            perturb = recompute = True
    return (perturb, recompute, extraprec, discard)