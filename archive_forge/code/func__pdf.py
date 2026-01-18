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