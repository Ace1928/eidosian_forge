from scipy import stats
from scipy.stats import distributions
import numpy as np
class Transf_gen(distributions.rv_continuous):
    """a class for non-linear monotonic transformation of a continuous random variable

    """

    def __init__(self, kls, func, funcinv, *args, **kwargs):
        self.func = func
        self.funcinv = funcinv
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.decr = kwargs.pop('decr', False)
        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls
        super().__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)

    def _cdf(self, x, *args, **kwargs):
        if not self.decr:
            return self.kls._cdf(self.funcinv(x), *args, **kwargs)
        else:
            return 1.0 - self.kls._cdf(self.funcinv(x), *args, **kwargs)

    def _ppf(self, q, *args, **kwargs):
        if not self.decr:
            return self.func(self.kls._ppf(q, *args, **kwargs))
        else:
            return self.func(self.kls._ppf(1 - q, *args, **kwargs))