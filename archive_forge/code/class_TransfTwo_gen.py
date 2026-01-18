from scipy import stats
from scipy.stats import distributions
import numpy as np
class TransfTwo_gen(distributions.rv_continuous):
    """Distribution based on a non-monotonic (u- or hump-shaped transformation)

    the constructor can be called with a distribution class, and functions
    that define the non-linear transformation.
    and generates the distribution of the transformed random variable

    Note: the transformation, it's inverse and derivatives need to be fully
    specified: func, funcinvplus, funcinvminus, derivplus,  derivminus.
    Currently no numerical derivatives or inverse are calculated

    This can be used to generate distribution instances similar to the
    distributions in scipy.stats.

    """

    def __init__(self, kls, func, funcinvplus, funcinvminus, derivplus, derivminus, *args, **kwargs):
        self.func = func
        self.funcinvplus = funcinvplus
        self.funcinvminus = funcinvminus
        self.derivplus = derivplus
        self.derivminus = derivminus
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.shape = kwargs.pop('shape', False)
        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls
        super().__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)

    def _rvs(self, *args):
        self.kls._size = self._size
        return self.func(self.kls._rvs(*args))

    def _pdf(self, x, *args, **kwargs):
        if self.shape == 'u':
            signpdf = 1
        elif self.shape == 'hump':
            signpdf = -1
        else:
            raise ValueError('shape can only be `u` or `hump`')
        return signpdf * (self.derivplus(x) * self.kls._pdf(self.funcinvplus(x), *args, **kwargs) - self.derivminus(x) * self.kls._pdf(self.funcinvminus(x), *args, **kwargs))

    def _cdf(self, x, *args, **kwargs):
        if self.shape == 'u':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
        else:
            return 1.0 - self._sf(x, *args, **kwargs)

    def _sf(self, x, *args, **kwargs):
        if self.shape == 'hump':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
        else:
            return 1.0 - self._cdf(x, *args, **kwargs)

    def _munp(self, n, *args, **kwargs):
        return self._mom0_sc(n, *args)