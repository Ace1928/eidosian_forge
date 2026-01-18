import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
class CubicRegressionSpline(object):
    """Base class for cubic regression spline stateful transforms

    This class contains all the functionality for the following stateful
    transforms:
     - ``cr(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for natural cubic regression spline
     - ``cc(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for cyclic cubic regression spline
    """
    common_doc = "\n    :arg df: The number of degrees of freedom to use for this spline. The\n      return value will have this many columns. You must specify at least one\n      of ``df`` and ``knots``.\n    :arg knots: The interior knots to use for the spline. If unspecified, then\n      equally spaced quantiles of the input data are used. You must specify at\n      least one of ``df`` and ``knots``.\n    :arg lower_bound: The lower exterior knot location.\n    :arg upper_bound: The upper exterior knot location.\n    :arg constraints: Either a 2-d array defining general linear constraints\n     (that is ``np.dot(constraints, betas)`` is zero, where ``betas`` denotes\n     the array of *initial* parameters, corresponding to the *initial*\n     unconstrained design matrix), or the string\n     ``'center'`` indicating that we should apply a centering constraint\n     (this constraint will be computed from the input data, remembered and\n     re-used for prediction from the fitted model).\n     The constraints are absorbed in the resulting design matrix which means\n     that the model is actually rewritten in terms of\n     *unconstrained* parameters. For more details see :ref:`spline-regression`.\n\n    This is a stateful transforms (for details see\n    :ref:`stateful-transforms`). If ``knots``, ``lower_bound``, or\n    ``upper_bound`` are not specified, they will be calculated from the data\n    and then the chosen values will be remembered and re-used for prediction\n    from the fitted model.\n\n    Using this function requires scipy be installed.\n\n    .. versionadded:: 0.3.0\n    "

    def __init__(self, name, cyclic):
        self._name = name
        self._cyclic = cyclic
        self._tmp = {}
        self._all_knots = None
        self._constraints = None

    def memorize_chunk(self, x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None):
        args = {'df': df, 'knots': knots, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'constraints': constraints}
        self._tmp['args'] = args
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError('Input to %r must be 1-d, or a 2-d column vector.' % (self._name,))
        self._tmp.setdefault('xs', []).append(x)

    def memorize_finish(self):
        args = self._tmp['args']
        xs = self._tmp['xs']
        del self._tmp
        x = np.concatenate(xs)
        if args['df'] is None and args['knots'] is None:
            raise ValueError("Must specify either 'df' or 'knots'.")
        constraints = args['constraints']
        n_constraints = 0
        if constraints is not None:
            if safe_string_eq(constraints, 'center'):
                n_constraints = 1
            else:
                constraints = np.atleast_2d(constraints)
                if constraints.ndim != 2:
                    raise ValueError('Constraints must be 2-d array or 1-d vector.')
                n_constraints = constraints.shape[0]
        n_inner_knots = None
        if args['df'] is not None:
            min_df = 1
            if not self._cyclic and n_constraints == 0:
                min_df = 2
            if args['df'] < min_df:
                raise ValueError("'df'=%r must be greater than or equal to %r." % (args['df'], min_df))
            n_inner_knots = args['df'] - 2 + n_constraints
            if self._cyclic:
                n_inner_knots += 1
        self._all_knots = _get_all_sorted_knots(x, n_inner_knots=n_inner_knots, inner_knots=args['knots'], lower_bound=args['lower_bound'], upper_bound=args['upper_bound'])
        if constraints is not None:
            if safe_string_eq(constraints, 'center'):
                constraints = _get_centering_constraint_from_dmatrix(_get_free_crs_dmatrix(x, self._all_knots, cyclic=self._cyclic))
            df_before_constraints = self._all_knots.size
            if self._cyclic:
                df_before_constraints -= 1
            if constraints.shape[1] != df_before_constraints:
                raise ValueError('Constraints array should have %r columns but %r found.' % (df_before_constraints, constraints.shape[1]))
            self._constraints = constraints

    def transform(self, x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None):
        x_orig = x
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError('Input to %r must be 1-d, or a 2-d column vector.' % (self._name,))
        dm = _get_crs_dmatrix(x, self._all_knots, self._constraints, cyclic=self._cyclic)
        if have_pandas:
            if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
                dm = pandas.DataFrame(dm)
                dm.index = x_orig.index
        return dm
    __getstate__ = no_pickling