from scipy import stats
from scipy.stats import distributions
import numpy as np
class ExpTransf_gen(distributions.rv_continuous):
    """Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    """

    def __init__(self, kls, *args, **kwargs):
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super().__init__(a=a, name=name)
        self.kls = kls

    def _cdf(self, x, *args):
        return self.kls._cdf(np.log(x), *args)

    def _ppf(self, q, *args):
        return np.exp(self.kls._ppf(q, *args))