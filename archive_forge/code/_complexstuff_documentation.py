complex-valued functions adapted from SciPy's cython code:

https://github.com/scipy/scipy/blob/master/scipy/special/_complexstuff.pxd

Notes:
- instead of zabs, use thrust::abs
- instead of zarg, use thrust::arg
- instead of zcos, use thrust::cos
- instead of zexp, use thrust::exp
- instead of zisfinite, use isfinite defined in _core/include/cupy/complex.cuh
- instead of zisinf, use isinf defined in _core/include/cupy/complex.cuh
- instead of zisnan, use isnan defined in _core/include/cupy/complex.cuh
- instead of zpack, use complex<double>(real, imag)
- instead of zpow, use thrust::pow
- instead of zreal, use z.real()
- instead of zsin, use thrust::sin
- instead of zsqrt, use thrust::sqrt

