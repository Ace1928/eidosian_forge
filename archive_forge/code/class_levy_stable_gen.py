import warnings
from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.integrate._quadrature import _builtincoeffs
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import scipy.special as sc
from scipy._lib._util import _lazywhere
from .._distn_infrastructure import rv_continuous, _ShapeInfo
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
from .levyst import Nolan
from scipy._lib.doccer import inherit_docstring_from
class levy_stable_gen(rv_continuous):
    """A Levy-stable continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy, levy_l, cauchy, norm

    Notes
    -----
    The distribution for `levy_stable` has characteristic function:

    .. math::

        \\varphi(t, \\alpha, \\beta, c, \\mu) =
        e^{it\\mu -|ct|^{\\alpha}(1-i\\beta\\operatorname{sign}(t)\\Phi(\\alpha, t))}

    where two different parameterizations are supported. The first :math:`S_1`:

    .. math::

        \\Phi = \\begin{cases}
                \\tan \\left({\\frac {\\pi \\alpha }{2}}\\right)&\\alpha \\neq 1\\\\
                -{\\frac {2}{\\pi }}\\log |t|&\\alpha =1
                \\end{cases}

    The second :math:`S_0`:

    .. math::

        \\Phi = \\begin{cases}
                -\\tan \\left({\\frac {\\pi \\alpha }{2}}\\right)(|ct|^{1-\\alpha}-1)
                &\\alpha \\neq 1\\\\
                -{\\frac {2}{\\pi }}\\log |ct|&\\alpha =1
                \\end{cases}


    The probability density function for `levy_stable` is:

    .. math::

        f(x) = \\frac{1}{2\\pi}\\int_{-\\infty}^\\infty \\varphi(t)e^{-ixt}\\,dt

    where :math:`-\\infty < t < \\infty`. This integral does not have a known
    closed form.

    `levy_stable` generalizes several distributions.  Where possible, they
    should be used instead.  Specifically, when the shape parameters
    assume the values in the table below, the corresponding equivalent
    distribution should be used.

    =========  ========  ===========
    ``alpha``  ``beta``   Equivalent
    =========  ========  ===========
     1/2       -1        `levy_l`
     1/2       1         `levy`
     1         0         `cauchy`
     2         any       `norm` (with ``scale=sqrt(2)``)
    =========  ========  ===========

    Evaluation of the pdf uses Nolan's piecewise integration approach with the
    Zolotarev :math:`M` parameterization by default. There is also the option
    to use direct numerical integration of the standard parameterization of the
    characteristic function or to evaluate by taking the FFT of the
    characteristic function.

    The default method can changed by setting the class variable
    ``levy_stable.pdf_default_method`` to one of 'piecewise' for Nolan's
    approach, 'dni' for direct numerical integration, or 'fft-simpson' for the
    FFT based approach. For the sake of backwards compatibility, the methods
    'best' and 'zolotarev' are equivalent to 'piecewise' and the method
    'quadrature' is equivalent to 'dni'.

    The parameterization can be changed  by setting the class variable
    ``levy_stable.parameterization`` to either 'S0' or 'S1'.
    The default is 'S1'.

    To improve performance of piecewise and direct numerical integration one
    can specify ``levy_stable.quad_eps`` (defaults to 1.2e-14). This is used
    as both the absolute and relative quadrature tolerance for direct numerical
    integration and as the relative quadrature tolerance for the piecewise
    method. One can also specify ``levy_stable.piecewise_x_tol_near_zeta``
    (defaults to 0.005) for how close x is to zeta before it is considered the
    same as x [NO]. The exact check is
    ``abs(x0 - zeta) < piecewise_x_tol_near_zeta*alpha**(1/alpha)``. One can
    also specify ``levy_stable.piecewise_alpha_tol_near_one`` (defaults to
    0.005) for how close alpha is to 1 before being considered equal to 1.

    To increase accuracy of FFT calculation one can specify
    ``levy_stable.pdf_fft_grid_spacing`` (defaults to 0.001) and
    ``pdf_fft_n_points_two_power`` (defaults to None which means a value is
    calculated that sufficiently covers the input range).

    Further control over FFT calculation is available by setting
    ``pdf_fft_interpolation_degree`` (defaults to 3) for spline order and
    ``pdf_fft_interpolation_level`` for determining the number of points to use
    in the Newton-Cotes formula when approximating the characteristic function
    (considered experimental).

    Evaluation of the cdf uses Nolan's piecewise integration approach with the
    Zolatarev :math:`S_0` parameterization by default. There is also the option
    to evaluate through integration of an interpolated spline of the pdf
    calculated by means of the FFT method. The settings affecting FFT
    calculation are the same as for pdf calculation. The default cdf method can
    be changed by setting ``levy_stable.cdf_default_method`` to either
    'piecewise' or 'fft-simpson'.  For cdf calculations the Zolatarev method is
    superior in accuracy, so FFT is disabled by default.

    Fitting estimate uses quantile estimation method in [MC]. MLE estimation of
    parameters in fit method uses this quantile estimate initially. Note that
    MLE doesn't always converge if using FFT for pdf calculations; this will be
    the case if alpha <= 1 where the FFT approach doesn't give good
    approximations.

    Any non-missing value for the attribute
    ``levy_stable.pdf_fft_min_points_threshold`` will set
    ``levy_stable.pdf_default_method`` to 'fft-simpson' if a valid
    default method is not otherwise set.



    .. warning::

        For pdf calculations FFT calculation is considered experimental.

        For cdf calculations FFT calculation is considered experimental. Use
        Zolatarev's method instead (default).

    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters.
    Generally ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
    equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc) / scale``, except in the ``S1`` parameterization if
    ``alpha == 1``.  In that case ``%(name)s.pdf(x, %(shapes)s, loc, scale)``
    is identically equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc - 2 * beta * scale * np.log(scale) / np.pi) / scale``.
    See [NO2]_ Definition 1.8 for more information.
    Note that shifting the location of a distribution
    does not make it a "noncentral" distribution.

    References
    ----------
    .. [MC] McCulloch, J., 1986. Simple consistent estimators of stable
        distribution parameters. Communications in Statistics - Simulation and
        Computation 15, 11091136.
    .. [WZ] Wang, Li and Zhang, Ji-Hong, 2008. Simpson's rule based FFT method
        to compute densities of stable distribution.
    .. [NO] Nolan, J., 1997. Numerical Calculation of Stable Densities and
        distributions Functions.
    .. [NO2] Nolan, J., 2018. Stable Distributions: Models for Heavy Tailed
        Data.
    .. [HO] Hopcraft, K. I., Jakeman, E., Tanner, R. M. J., 1999. Lévy random
        walks with fluctuating step number and multiscale behavior.

    %(example)s

    """
    parameterization = 'S1'
    pdf_default_method = 'piecewise'
    cdf_default_method = 'piecewise'
    quad_eps = _QUAD_EPS
    piecewise_x_tol_near_zeta = 0.005
    piecewise_alpha_tol_near_one = 0.005
    pdf_fft_min_points_threshold = None
    pdf_fft_grid_spacing = 0.001
    pdf_fft_n_points_two_power = None
    pdf_fft_interpolation_level = 3
    pdf_fft_interpolation_degree = 3

    def _argcheck(self, alpha, beta):
        return (alpha > 0) & (alpha <= 2) & (beta <= 1) & (beta >= -1)

    def _shape_info(self):
        ialpha = _ShapeInfo('alpha', False, (0, 2), (False, True))
        ibeta = _ShapeInfo('beta', False, (-1, 1), (True, True))
        return [ialpha, ibeta]

    def _parameterization(self):
        allowed = ('S0', 'S1')
        pz = self.parameterization
        if pz not in allowed:
            raise RuntimeError(f"Parameterization '{pz}' in supported list: {allowed}")
        return pz

    @inherit_docstring_from(rv_continuous)
    def rvs(self, *args, **kwds):
        X1 = super().rvs(*args, **kwds)
        kwds.pop('discrete', None)
        kwds.pop('random_state', None)
        (alpha, beta), delta, gamma, size = self._parse_args_rvs(*args, **kwds)
        X1 = np.where(alpha == 1.0, X1 + 2 * beta * gamma * np.log(gamma) / np.pi, X1)
        if self._parameterization() == 'S0':
            return np.where(alpha == 1.0, X1 - beta * 2 * gamma * np.log(gamma) / np.pi, X1 - gamma * beta * np.tan(np.pi * alpha / 2.0))
        elif self._parameterization() == 'S1':
            return X1

    def _rvs(self, alpha, beta, size=None, random_state=None):
        return _rvs_Z1(alpha, beta, size, random_state)

    @inherit_docstring_from(rv_continuous)
    def pdf(self, x, *args, **kwds):
        if self._parameterization() == 'S0':
            return super().pdf(x, *args, **kwds)
        elif self._parameterization() == 'S1':
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                return super().pdf(x, *args, **kwds)
            else:
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
                data_in = np.dstack((x, alpha, beta))[0]
                data_out = np.empty(shape=(len(data_in), 1))
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = delta + 2 * _beta * gamma * np.log(gamma) / np.pi if _alpha == 1.0 else delta
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = super().pdf(_x, _alpha, _beta, loc=_delta, scale=gamma).reshape(len(_x), 1)
                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output

    def _pdf(self, x, alpha, beta):
        if self._parameterization() == 'S0':
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z0
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z0
            _cf = _cf_Z0
        elif self._parameterization() == 'S1':
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z1
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z1
            _cf = _cf_Z1
        x = np.asarray(x).reshape(1, -1)[0, :]
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
        data_in = np.dstack((x, alpha, beta))[0]
        data_out = np.empty(shape=(len(data_in), 1))
        pdf_default_method_name = self.pdf_default_method
        if pdf_default_method_name in ('piecewise', 'best', 'zolotarev'):
            pdf_single_value_method = _pdf_single_value_piecewise
        elif pdf_default_method_name in ('dni', 'quadrature'):
            pdf_single_value_method = _pdf_single_value_cf_integrate
        elif pdf_default_method_name == 'fft-simpson' or self.pdf_fft_min_points_threshold is not None:
            pdf_single_value_method = None
        pdf_single_value_kwds = {'quad_eps': self.quad_eps, 'piecewise_x_tol_near_zeta': self.piecewise_x_tol_near_zeta, 'piecewise_alpha_tol_near_one': self.piecewise_alpha_tol_near_one}
        fft_grid_spacing = self.pdf_fft_grid_spacing
        fft_n_points_two_power = self.pdf_fft_n_points_two_power
        fft_interpolation_level = self.pdf_fft_interpolation_level
        fft_interpolation_degree = self.pdf_fft_interpolation_degree
        uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
        for pair in uniq_param_pairs:
            data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
            data_subset = data_in[data_mask]
            if pdf_single_value_method is not None:
                data_out[data_mask] = np.array([pdf_single_value_method(_x, _alpha, _beta, **pdf_single_value_kwds) for _x, _alpha, _beta in data_subset]).reshape(len(data_subset), 1)
            else:
                warnings.warn('Density calculations experimental for FFT method.' + ' Use combination of piecewise and dni methods instead.', RuntimeWarning, stacklevel=3)
                _alpha, _beta = pair
                _x = data_subset[:, (0,)]
                if _alpha < 1.0:
                    raise RuntimeError('FFT method does not work well for alpha less than 1.')
                if fft_grid_spacing is None and fft_n_points_two_power is None:
                    raise ValueError('One of fft_grid_spacing or fft_n_points_two_power ' + 'needs to be set.')
                max_abs_x = np.max(np.abs(_x))
                h = 2 ** (3 - fft_n_points_two_power) * max_abs_x if fft_grid_spacing is None else fft_grid_spacing
                q = np.ceil(np.log(2 * max_abs_x / h) / np.log(2)) + 2 if fft_n_points_two_power is None else int(fft_n_points_two_power)
                MAX_Q = 30
                if q > MAX_Q:
                    raise RuntimeError('fft_n_points_two_power has a maximum ' + f'value of {MAX_Q}')
                density_x, density = pdf_from_cf_with_fft(lambda t: _cf(t, _alpha, _beta), h=h, q=q, level=fft_interpolation_level)
                f = interpolate.InterpolatedUnivariateSpline(density_x, np.real(density), k=fft_interpolation_degree)
                data_out[data_mask] = f(_x)
        return data_out.T[0]

    @inherit_docstring_from(rv_continuous)
    def cdf(self, x, *args, **kwds):
        if self._parameterization() == 'S0':
            return super().cdf(x, *args, **kwds)
        elif self._parameterization() == 'S1':
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                return super().cdf(x, *args, **kwds)
            else:
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
                data_in = np.dstack((x, alpha, beta))[0]
                data_out = np.empty(shape=(len(data_in), 1))
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = delta + 2 * _beta * gamma * np.log(gamma) / np.pi if _alpha == 1.0 else delta
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = super().cdf(_x, _alpha, _beta, loc=_delta, scale=gamma).reshape(len(_x), 1)
                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output

    def _cdf(self, x, alpha, beta):
        if self._parameterization() == 'S0':
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z0
            _cf = _cf_Z0
        elif self._parameterization() == 'S1':
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z1
            _cf = _cf_Z1
        x = np.asarray(x).reshape(1, -1)[0, :]
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
        data_in = np.dstack((x, alpha, beta))[0]
        data_out = np.empty(shape=(len(data_in), 1))
        cdf_default_method_name = self.cdf_default_method
        if cdf_default_method_name == 'piecewise':
            cdf_single_value_method = _cdf_single_value_piecewise
        elif cdf_default_method_name == 'fft-simpson':
            cdf_single_value_method = None
        cdf_single_value_kwds = {'quad_eps': self.quad_eps, 'piecewise_x_tol_near_zeta': self.piecewise_x_tol_near_zeta, 'piecewise_alpha_tol_near_one': self.piecewise_alpha_tol_near_one}
        fft_grid_spacing = self.pdf_fft_grid_spacing
        fft_n_points_two_power = self.pdf_fft_n_points_two_power
        fft_interpolation_level = self.pdf_fft_interpolation_level
        fft_interpolation_degree = self.pdf_fft_interpolation_degree
        uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
        for pair in uniq_param_pairs:
            data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
            data_subset = data_in[data_mask]
            if cdf_single_value_method is not None:
                data_out[data_mask] = np.array([cdf_single_value_method(_x, _alpha, _beta, **cdf_single_value_kwds) for _x, _alpha, _beta in data_subset]).reshape(len(data_subset), 1)
            else:
                warnings.warn('Cumulative density calculations experimental for FFT' + ' method. Use piecewise method instead.', RuntimeWarning, stacklevel=3)
                _alpha, _beta = pair
                _x = data_subset[:, (0,)]
                if fft_grid_spacing is None and fft_n_points_two_power is None:
                    raise ValueError('One of fft_grid_spacing or fft_n_points_two_power ' + 'needs to be set.')
                max_abs_x = np.max(np.abs(_x))
                h = 2 ** (3 - fft_n_points_two_power) * max_abs_x if fft_grid_spacing is None else fft_grid_spacing
                q = np.ceil(np.log(2 * max_abs_x / h) / np.log(2)) + 2 if fft_n_points_two_power is None else int(fft_n_points_two_power)
                density_x, density = pdf_from_cf_with_fft(lambda t: _cf(t, _alpha, _beta), h=h, q=q, level=fft_interpolation_level)
                f = interpolate.InterpolatedUnivariateSpline(density_x, np.real(density), k=fft_interpolation_degree)
                data_out[data_mask] = np.array([f.integral(self.a, float(x_1.squeeze())) for x_1 in _x]).reshape(data_out[data_mask].shape)
        return data_out.T[0]

    def _fitstart(self, data):
        if self._parameterization() == 'S0':
            _fitstart = _fitstart_S0
        elif self._parameterization() == 'S1':
            _fitstart = _fitstart_S1
        return _fitstart(data)

    def _stats(self, alpha, beta):
        mu = 0 if alpha > 1 else np.nan
        mu2 = 2 if alpha == 2 else np.inf
        g1 = 0.0 if alpha == 2.0 else np.nan
        g2 = 0.0 if alpha == 2.0 else np.nan
        return (mu, mu2, g1, g2)