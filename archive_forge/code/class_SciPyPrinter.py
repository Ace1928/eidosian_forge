from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
class SciPyPrinter(NumPyPrinter):
    _kf = {**NumPyPrinter._kf, **_scipy_known_functions}
    _kc = {**NumPyPrinter._kc, **_scipy_known_constants}

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.language = 'Python with SciPy and NumPy'

    def _print_SparseRepMatrix(self, expr):
        i, j, data = ([], [], [])
        for (r, c), v in expr.todok().items():
            i.append(r)
            j.append(c)
            data.append(v)
        return '{name}(({data}, ({i}, {j})), shape={shape})'.format(name=self._module_format('scipy.sparse.coo_matrix'), data=data, i=i, j=j, shape=expr.shape)
    _print_ImmutableSparseMatrix = _print_SparseRepMatrix

    def _print_assoc_legendre(self, expr):
        return '{0}({2}, {1}, {3})'.format(self._module_format('scipy.special.lpmv'), self._print(expr.args[0]), self._print(expr.args[1]), self._print(expr.args[2]))

    def _print_lowergamma(self, expr):
        return '{0}({2})*{1}({2}, {3})'.format(self._module_format('scipy.special.gamma'), self._module_format('scipy.special.gammainc'), self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_uppergamma(self, expr):
        return '{0}({2})*{1}({2}, {3})'.format(self._module_format('scipy.special.gamma'), self._module_format('scipy.special.gammaincc'), self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_betainc(self, expr):
        betainc = self._module_format('scipy.special.betainc')
        beta = self._module_format('scipy.special.beta')
        args = [self._print(arg) for arg in expr.args]
        return f'({betainc}({args[0]}, {args[1]}, {args[3]}) - {betainc}({args[0]}, {args[1]}, {args[2]}))             * {beta}({args[0]}, {args[1]})'

    def _print_betainc_regularized(self, expr):
        return '{0}({1}, {2}, {4}) - {0}({1}, {2}, {3})'.format(self._module_format('scipy.special.betainc'), self._print(expr.args[0]), self._print(expr.args[1]), self._print(expr.args[2]), self._print(expr.args[3]))

    def _print_fresnels(self, expr):
        return '{}({})[0]'.format(self._module_format('scipy.special.fresnel'), self._print(expr.args[0]))

    def _print_fresnelc(self, expr):
        return '{}({})[1]'.format(self._module_format('scipy.special.fresnel'), self._print(expr.args[0]))

    def _print_airyai(self, expr):
        return '{}({})[0]'.format(self._module_format('scipy.special.airy'), self._print(expr.args[0]))

    def _print_airyaiprime(self, expr):
        return '{}({})[1]'.format(self._module_format('scipy.special.airy'), self._print(expr.args[0]))

    def _print_airybi(self, expr):
        return '{}({})[2]'.format(self._module_format('scipy.special.airy'), self._print(expr.args[0]))

    def _print_airybiprime(self, expr):
        return '{}({})[3]'.format(self._module_format('scipy.special.airy'), self._print(expr.args[0]))

    def _print_bernoulli(self, expr):
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_harmonic(self, expr):
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)
        if len(limits) == 1:
            module_str = self._module_format('scipy.integrate.quad')
            limit_str = '%s, %s' % tuple(map(self._print, limits[0]))
        else:
            module_str = self._module_format('scipy.integrate.nquad')
            limit_str = '({})'.format(', '.join(('(%s, %s)' % tuple(map(self._print, l)) for l in limits)))
        return '{}(lambda {}: {}, {})[0]'.format(module_str, ', '.join(map(self._print, integration_vars)), self._print(e.args[0]), limit_str)

    def _print_Si(self, expr):
        return '{}({})[0]'.format(self._module_format('scipy.special.sici'), self._print(expr.args[0]))

    def _print_Ci(self, expr):
        return '{}({})[1]'.format(self._module_format('scipy.special.sici'), self._print(expr.args[0]))