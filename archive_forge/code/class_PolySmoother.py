import numpy as np
from . import kernels
class PolySmoother:
    """
    Polynomial smoother up to a given order.
    Fit based on weighted least squares.

    The x values can be specified at instantiation or when called.

    This is a 3 liner with OLS or WLS, see test.
    It's here as a test smoother for GAM
    """

    def __init__(self, order, x=None):
        self.order = order
        self.coef = np.zeros((order + 1,), np.float64)
        if x is not None:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother init, shape:', x.shape)
                x = x[0, :]
            self.X = np.array([x ** i for i in range(order + 1)]).T

    def df_fit(self):
        """alias of df_model for backwards compatibility
        """
        return self.df_model()

    def df_model(self):
        """
        Degrees of freedom used in the fit.
        """
        return self.order + 1

    def gram(self, d=None):
        pass

    def smooth(self, *args, **kwds):
        """alias for fit,  for backwards compatibility,

        do we need it with different behavior than fit?

        """
        return self.fit(*args, **kwds)

    def df_resid(self):
        """
        Residual degrees of freedom from last fit.
        """
        return self.N - self.order - 1

    def __call__(self, x=None):
        return self.predict(x=x)

    def predict(self, x=None):
        if x is not None:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother predict, shape:', x.shape)
                x = x[:, 0]
            X = np.array([x ** i for i in range(self.order + 1)])
        else:
            X = self.X
        if X.shape[1] == self.coef.shape[0]:
            return np.squeeze(np.dot(X, self.coef))
        else:
            return np.squeeze(np.dot(X.T, self.coef))

    def fit(self, y, x=None, weights=None):
        self.N = y.shape[0]
        if y.ndim == 1:
            y = y[:, None]
        if weights is None or np.isnan(weights).all():
            weights = 1
            _w = 1
        else:
            _w = np.sqrt(weights)[:, None]
        if x is None:
            if not hasattr(self, 'X'):
                raise ValueError('x needed to fit PolySmoother')
        else:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother predict, shape:', x.shape)
            self.X = np.array([x ** i for i in range(self.order + 1)]).T
        X = self.X * _w
        _y = y * _w
        self.coef = np.linalg.lstsq(X, _y, rcond=-1)[0]
        self.params = np.squeeze(self.coef)