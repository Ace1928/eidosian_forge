from math import prod
from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
from sympy.external import import_module
class MatrixDistribution(Distribution, NamedArgsMixin):
    """
    Abstract class for Matrix Distribution.
    """

    def __new__(cls, *args):
        args = [ImmutableMatrix(arg) if isinstance(arg, list) else _sympify(arg) for arg in args]
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    def __call__(self, expr):
        if isinstance(expr, list):
            expr = ImmutableMatrix(expr)
        return self.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        libraries = ['scipy', 'numpy', 'pymc3', 'pymc']
        if library not in libraries:
            raise NotImplementedError('Sampling from %s is not supported yet.' % str(library))
        if not import_module(library):
            raise ValueError('Failed to import %s' % library)
        samps = _get_sample_class_matrixrv[library](self, size, seed)
        if samps is not None:
            return samps
        raise NotImplementedError('Sampling for %s is not currently implemented from %s' % (self.__class__.__name__, library))