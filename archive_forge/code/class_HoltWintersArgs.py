import numpy as np
class HoltWintersArgs:

    def __init__(self, xi, p, bounds, y, m, n, transform=False):
        self._xi = xi
        self._p = p
        self._bounds = bounds
        self._y = y
        self._lvl = np.empty(n)
        self._b = np.empty(n)
        self._s = np.empty(n + m - 1)
        self._m = m
        self._n = n
        self._transform = transform

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, value):
        self._xi = value

    @property
    def p(self):
        return self._p

    @property
    def bounds(self):
        return self._bounds

    @property
    def y(self):
        return self._y

    @property
    def lvl(self):
        return self._lvl

    @property
    def b(self):
        return self._b

    @property
    def s(self):
        return self._s

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value