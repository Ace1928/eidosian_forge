import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    Examples
    --------
    >>> x = np.linspace(1, 2, 25)
    >>> y = 1.5 * x + 3
    >>> lm = LineModelND()
    >>> lm.estimate(np.stack([x, y], axis=-1))
    True
    >>> tuple(np.round(lm.params, 5))
    (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    >>> res = lm.residuals(np.stack([x, y], axis=-1))
    >>> np.abs(np.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    >>> np.round(lm.predict_y(x[:5]), 3)
    array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    >>> np.round(lm.predict_x(y[:5]), 3)
    array([1.   , 1.042, 1.083, 1.125, 1.167])

    """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)
        origin = data.mean(axis=0)
        data = data - origin
        if data.shape[0] == 2:
            direction = data[1] - data[0]
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
        elif data.shape[0] > 2:
            _, _, v = np.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:
            return False
        self.params = (origin, direction)
        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N,) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')
        origin, direction = params
        res = data - origin - ((data - origin) @ direction)[..., np.newaxis] * direction
        return np.linalg.norm(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')
        origin, direction = params
        if direction[axis] == 0:
            raise ValueError(f'Line parallel to axis {axis}')
        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data

    def predict_x(self, y, params=None):
        """Predict x-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(y, axis=1)[:, 0]

        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        x : array
            Predicted x-coordinates.

        """
        x = self.predict(y, axis=1, params=params)[:, 0]
        return x

    def predict_y(self, x, params=None):
        """Predict y-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(x, axis=0)[:, 1]

        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        y : array
            Predicted y-coordinates.

        """
        y = self.predict(x, axis=0, params=params)[:, 1]
        return y