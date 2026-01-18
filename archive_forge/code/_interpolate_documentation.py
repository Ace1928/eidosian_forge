import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np

        Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on ``[xa, xb]`` and having the values and derivatives at the
        endpoints `xa` and `xb` as specified by `ya`` and `yb`.

        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of `ya` and `yb` are `na` and `nb`, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at `xa`. `ya[0]` is the value of the function, and
            `ya[i]` for ``i > 0`` is the value of the ``i``th derivative.
        yb : array_like
            Derivatives at `xb`.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

            .. math:: Q_a = \sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At ``x = xb`` it's the same with ``a = n - q``.
        