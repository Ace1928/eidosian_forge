import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _ndtri(p):
    """Implements ndtri core logic."""
    p0 = [-1.2391658386738125, 13.931260938727968, -56.67628574690703, 98.00107541859997, -59.96335010141079]
    q0 = [-1.1833162112133, 15.90562251262117, -82.03722561683334, 200.26021238006066, -225.46268785411937, 86.36024213908905, 4.676279128988815, 1.9544885833814176, 1.0]
    p1 = [-0.0008574567851546854, -0.03504246268278482, -0.1402560791713545, 2.1866330685079025, 14.684956192885803, 44.08050738932008, 57.16281922464213, 31.525109459989388, 4.0554489230596245]
    q1 = [-0.0009332594808954574, -0.03808064076915783, -0.14218292285478779, 2.504649462083094, 15.04253856929075, 41.3172038254672, 45.39076351288792, 15.779988325646675, 1.0]
    p2 = [6.239745391849833e-09, 2.6580697468673755e-06, 0.00030158155350823543, 0.012371663481782003, 0.20148538954917908, 1.3330346081580755, 3.9388102529247444, 6.915228890689842, 3.2377489177694603]
    q2 = [6.790194080099813e-09, 2.8924786474538068e-06, 0.00032801446468212774, 0.013420400608854318, 0.21623699359449663, 1.3770209948908132, 3.6798356385616087, 6.02427039364742, 1.0]

    def _create_polynomial(var, coeffs):
        """Compute n_th order polynomial via Horner's method."""
        coeffs = np.array(coeffs, var.dtype.as_numpy_dtype)
        if not coeffs.size:
            return array_ops.zeros_like(var)
        return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var
    maybe_complement_p = array_ops.where_v2(p > -np.expm1(-2.0), 1.0 - p, p)
    sanitized_mcp = array_ops.where_v2(maybe_complement_p <= 0.0, array_ops.fill(array_ops.shape(p), np.array(0.5, p.dtype.as_numpy_dtype)), maybe_complement_p)
    w = sanitized_mcp - 0.5
    ww = w ** 2
    x_for_big_p = w + w * ww * (_create_polynomial(ww, p0) / _create_polynomial(ww, q0))
    x_for_big_p *= -np.sqrt(2.0 * np.pi)
    z = math_ops.sqrt(-2.0 * math_ops.log(sanitized_mcp))
    first_term = z - math_ops.log(z) / z
    second_term_small_p = _create_polynomial(1.0 / z, p2) / _create_polynomial(1.0 / z, q2) / z
    second_term_otherwise = _create_polynomial(1.0 / z, p1) / _create_polynomial(1.0 / z, q1) / z
    x_for_small_p = first_term - second_term_small_p
    x_otherwise = first_term - second_term_otherwise
    x = array_ops.where_v2(sanitized_mcp > np.exp(-2.0), x_for_big_p, array_ops.where_v2(z >= 8.0, x_for_small_p, x_otherwise))
    x = array_ops.where_v2(p > 1.0 - np.exp(-2.0), x, -x)
    infinity_scalar = constant_op.constant(np.inf, dtype=p.dtype)
    infinity = array_ops.fill(array_ops.shape(p), infinity_scalar)
    x_nan_replaced = array_ops.where_v2(p <= 0.0, -infinity, array_ops.where_v2(p >= 1.0, infinity, x))
    return x_nan_replaced