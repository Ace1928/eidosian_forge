import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
class NormExpan_gen(distributions.rv_continuous):
    """Gram-Charlier Expansion of Normal distribution

    class follows scipy.stats.distributions pattern
    but with __init__

    """

    def __init__(self, args, **kwds):
        distributions.rv_continuous.__init__(self, name='Normal Expansion distribution', shapes=' ')
        mode = kwds.get('mode', 'sample')
        if mode == 'sample':
            mu, sig, sk, kur = stats.describe(args)[2:]
            self.mvsk = (mu, sig, sk, kur)
            cnt = mvsk2mc((mu, sig, sk, kur))
        elif mode == 'mvsk':
            cnt = mvsk2mc(args)
            self.mvsk = args
        elif mode == 'centmom':
            cnt = args
            self.mvsk = mc2mvsk(cnt)
        else:
            raise ValueError("mode must be 'mvsk' or centmom")
        self.cnt = cnt
        self._pdf = pdf_mvsk(self.mvsk)

    def _munp(self, n):
        return self._mom0_sc(n)

    def _stats_skip(self):
        return self.mvsk