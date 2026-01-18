import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
class KalmanFilter(Representation):
    """
    State space representation of a time series process, with Kalman filter

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the transition equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    loglikelihood_burn : int, optional
        The number of initial periods during which the loglikelihood is not
        recorded. Default is 0.
    tolerance : float, optional
        The tolerance at which the Kalman filter determines convergence to
        steady-state. Default is 1e-19.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `FilterResults`. If specified, class must extend from `FilterResults`.
    **kwargs
        Keyword arguments may be used to provide values for the filter,
        inversion, and stability methods. See `set_filter_method`,
        `set_inversion_method`, and `set_stability_method`.
        Keyword arguments may be used to provide default values for state space
        matrices. See `Representation` for more details.

    See Also
    --------
    FilterResults
    statsmodels.tsa.statespace.representation.Representation

    Notes
    -----
    There are several types of options available for controlling the Kalman
    filter operation. All options are internally held as bitmasks, but can be
    manipulated by setting class attributes, which act like boolean flags. For
    more information, see the `set_*` class method documentation. The options
    are:

    filter_method
        The filtering method controls aspects of which
        Kalman filtering approach will be used.
    inversion_method
        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.
    stability_method
        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.
    conserve_memory
        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.
    filter_timing
        By default, the Kalman filter follows Durbin and Koopman, 2012, in
        initializing the filter with predicted values. Kim and Nelson, 1999,
        instead initialize the filter with filtered values, which is
        essentially just a different timing convention.

    The `filter_method` and `inversion_method` options intentionally allow
    the possibility that multiple methods will be indicated. In the case that
    multiple methods are selected, the underlying Kalman filter will attempt to
    select the optional method given the input data.

    For example, it may be that INVERT_UNIVARIATE and SOLVE_CHOLESKY are
    indicated (this is in fact the default case). In this case, if the
    endogenous vector is 1-dimensional (`k_endog` = 1), then INVERT_UNIVARIATE
    is used and inversion reduces to simple division, and if it has a larger
    dimension, the Cholesky decomposition along with linear solving (rather
    than explicit matrix inversion) is used. If only SOLVE_CHOLESKY had been
    set, then the Cholesky decomposition method would *always* be used, even in
    the case of 1-dimensional data.
    """
    filter_methods = ['filter_conventional', 'filter_exact_initial', 'filter_augmented', 'filter_square_root', 'filter_univariate', 'filter_collapsed', 'filter_extended', 'filter_unscented', 'filter_concentrated', 'filter_chandrasekhar']
    filter_conventional = OptionWrapper('filter_method', FILTER_CONVENTIONAL)
    '\n    (bool) Flag for conventional Kalman filtering.\n    '
    filter_exact_initial = OptionWrapper('filter_method', FILTER_EXACT_INITIAL)
    '\n    (bool) Flag for exact initial Kalman filtering. Not implemented.\n    '
    filter_augmented = OptionWrapper('filter_method', FILTER_AUGMENTED)
    '\n    (bool) Flag for augmented Kalman filtering. Not implemented.\n    '
    filter_square_root = OptionWrapper('filter_method', FILTER_SQUARE_ROOT)
    '\n    (bool) Flag for square-root Kalman filtering. Not implemented.\n    '
    filter_univariate = OptionWrapper('filter_method', FILTER_UNIVARIATE)
    '\n    (bool) Flag for univariate filtering of multivariate observation vector.\n    '
    filter_collapsed = OptionWrapper('filter_method', FILTER_COLLAPSED)
    '\n    (bool) Flag for Kalman filtering with collapsed observation vector.\n    '
    filter_extended = OptionWrapper('filter_method', FILTER_EXTENDED)
    '\n    (bool) Flag for extended Kalman filtering. Not implemented.\n    '
    filter_unscented = OptionWrapper('filter_method', FILTER_UNSCENTED)
    '\n    (bool) Flag for unscented Kalman filtering. Not implemented.\n    '
    filter_concentrated = OptionWrapper('filter_method', FILTER_CONCENTRATED)
    '\n    (bool) Flag for Kalman filtering with concentrated log-likelihood.\n    '
    filter_chandrasekhar = OptionWrapper('filter_method', FILTER_CHANDRASEKHAR)
    '\n    (bool) Flag for filtering with Chandrasekhar recursions.\n    '
    inversion_methods = ['invert_univariate', 'solve_lu', 'invert_lu', 'solve_cholesky', 'invert_cholesky']
    invert_univariate = OptionWrapper('inversion_method', INVERT_UNIVARIATE)
    '\n    (bool) Flag for univariate inversion method (recommended).\n    '
    solve_lu = OptionWrapper('inversion_method', SOLVE_LU)
    '\n    (bool) Flag for LU and linear solver inversion method.\n    '
    invert_lu = OptionWrapper('inversion_method', INVERT_LU)
    '\n    (bool) Flag for LU inversion method.\n    '
    solve_cholesky = OptionWrapper('inversion_method', SOLVE_CHOLESKY)
    '\n    (bool) Flag for Cholesky and linear solver inversion method (recommended).\n    '
    invert_cholesky = OptionWrapper('inversion_method', INVERT_CHOLESKY)
    '\n    (bool) Flag for Cholesky inversion method.\n    '
    stability_methods = ['stability_force_symmetry']
    stability_force_symmetry = OptionWrapper('stability_method', STABILITY_FORCE_SYMMETRY)
    '\n    (bool) Flag for enforcing covariance matrix symmetry\n    '
    memory_options = ['memory_store_all', 'memory_no_forecast_mean', 'memory_no_forecast_cov', 'memory_no_forecast', 'memory_no_predicted_mean', 'memory_no_predicted_cov', 'memory_no_predicted', 'memory_no_filtered_mean', 'memory_no_filtered_cov', 'memory_no_filtered', 'memory_no_likelihood', 'memory_no_gain', 'memory_no_smoothing', 'memory_no_std_forecast', 'memory_conserve']
    memory_store_all = OptionWrapper('conserve_memory', MEMORY_STORE_ALL)
    '\n    (bool) Flag for storing all intermediate results in memory (default).\n    '
    memory_no_forecast_mean = OptionWrapper('conserve_memory', MEMORY_NO_FORECAST_MEAN)
    '\n    (bool) Flag to prevent storing forecasts and forecast errors.\n    '
    memory_no_forecast_cov = OptionWrapper('conserve_memory', MEMORY_NO_FORECAST_COV)
    '\n    (bool) Flag to prevent storing forecast error covariance matrices.\n    '

    @property
    def memory_no_forecast(self):
        """
        (bool) Flag to prevent storing all forecast-related output.
        """
        return self.memory_no_forecast_mean or self.memory_no_forecast_cov

    @memory_no_forecast.setter
    def memory_no_forecast(self, value):
        if bool(value):
            self.memory_no_forecast_mean = True
            self.memory_no_forecast_cov = True
        else:
            self.memory_no_forecast_mean = False
            self.memory_no_forecast_cov = False
    memory_no_predicted_mean = OptionWrapper('conserve_memory', MEMORY_NO_PREDICTED_MEAN)
    '\n    (bool) Flag to prevent storing predicted states.\n    '
    memory_no_predicted_cov = OptionWrapper('conserve_memory', MEMORY_NO_PREDICTED_COV)
    '\n    (bool) Flag to prevent storing predicted state covariance matrices.\n    '

    @property
    def memory_no_predicted(self):
        """
        (bool) Flag to prevent storing predicted state and covariance matrices.
        """
        return self.memory_no_predicted_mean or self.memory_no_predicted_cov

    @memory_no_predicted.setter
    def memory_no_predicted(self, value):
        if bool(value):
            self.memory_no_predicted_mean = True
            self.memory_no_predicted_cov = True
        else:
            self.memory_no_predicted_mean = False
            self.memory_no_predicted_cov = False
    memory_no_filtered_mean = OptionWrapper('conserve_memory', MEMORY_NO_FILTERED_MEAN)
    '\n    (bool) Flag to prevent storing filtered states.\n    '
    memory_no_filtered_cov = OptionWrapper('conserve_memory', MEMORY_NO_FILTERED_COV)
    '\n    (bool) Flag to prevent storing filtered state covariance matrices.\n    '

    @property
    def memory_no_filtered(self):
        """
        (bool) Flag to prevent storing filtered state and covariance matrices.
        """
        return self.memory_no_filtered_mean or self.memory_no_filtered_cov

    @memory_no_filtered.setter
    def memory_no_filtered(self, value):
        if bool(value):
            self.memory_no_filtered_mean = True
            self.memory_no_filtered_cov = True
        else:
            self.memory_no_filtered_mean = False
            self.memory_no_filtered_cov = False
    memory_no_likelihood = OptionWrapper('conserve_memory', MEMORY_NO_LIKELIHOOD)
    '\n    (bool) Flag to prevent storing likelihood values for each observation.\n    '
    memory_no_gain = OptionWrapper('conserve_memory', MEMORY_NO_GAIN)
    '\n    (bool) Flag to prevent storing the Kalman gain matrices.\n    '
    memory_no_smoothing = OptionWrapper('conserve_memory', MEMORY_NO_SMOOTHING)
    '\n    (bool) Flag to prevent storing likelihood values for each observation.\n    '
    memory_no_std_forecast = OptionWrapper('conserve_memory', MEMORY_NO_STD_FORECAST)
    '\n    (bool) Flag to prevent storing standardized forecast errors.\n    '
    memory_conserve = OptionWrapper('conserve_memory', MEMORY_CONSERVE)
    '\n    (bool) Flag to conserve the maximum amount of memory.\n    '
    timing_options = ['timing_init_predicted', 'timing_init_filtered']
    timing_init_predicted = OptionWrapper('filter_timing', TIMING_INIT_PREDICTED)
    '\n    (bool) Flag for the default timing convention (Durbin and Koopman, 2012).\n    '
    timing_init_filtered = OptionWrapper('filter_timing', TIMING_INIT_FILTERED)
    '\n    (bool) Flag for the alternate timing convention (Kim and Nelson, 2012).\n    '
    filter_method = FILTER_CONVENTIONAL
    '\n    (int) Filtering method bitmask.\n    '
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    '\n    (int) Inversion method bitmask.\n    '
    stability_method = STABILITY_FORCE_SYMMETRY
    '\n    (int) Stability method bitmask.\n    '
    conserve_memory = MEMORY_STORE_ALL
    '\n    (int) Memory conservation bitmask.\n    '
    filter_timing = TIMING_INIT_PREDICTED
    '\n    (int) Filter timing.\n    '

    def __init__(self, k_endog, k_states, k_posdef=None, loglikelihood_burn=0, tolerance=1e-19, results_class=None, kalman_filter_classes=None, **kwargs):
        keys = ['filter_method'] + KalmanFilter.filter_methods
        filter_method_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['inversion_method'] + KalmanFilter.inversion_methods
        inversion_method_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['stability_method'] + KalmanFilter.stability_methods
        stability_method_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['conserve_memory'] + KalmanFilter.memory_options
        conserve_memory_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['alternate_timing'] + KalmanFilter.timing_options
        filter_timing_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        super().__init__(k_endog, k_states, k_posdef, **kwargs)
        self._kalman_filters = {}
        self.loglikelihood_burn = loglikelihood_burn
        self.results_class = results_class if results_class is not None else FilterResults
        self.prefix_kalman_filter_map = kalman_filter_classes if kalman_filter_classes is not None else tools.prefix_kalman_filter_map.copy()
        self.set_filter_method(**filter_method_kwargs)
        self.set_inversion_method(**inversion_method_kwargs)
        self.set_stability_method(**stability_method_kwargs)
        self.set_conserve_memory(**conserve_memory_kwargs)
        self.set_filter_timing(**filter_timing_kwargs)
        self.tolerance = tolerance
        self._scale = None

    def _clone_kwargs(self, endog, **kwargs):
        kwargs = super()._clone_kwargs(endog, **kwargs)
        kwargs.setdefault('filter_method', self.filter_method)
        kwargs.setdefault('inversion_method', self.inversion_method)
        kwargs.setdefault('stability_method', self.stability_method)
        kwargs.setdefault('conserve_memory', self.conserve_memory)
        kwargs.setdefault('alternate_timing', bool(self.filter_timing))
        kwargs.setdefault('tolerance', self.tolerance)
        kwargs.setdefault('loglikelihood_burn', self.loglikelihood_burn)
        return kwargs

    @property
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    def _initialize_filter(self, filter_method=None, inversion_method=None, stability_method=None, conserve_memory=None, tolerance=None, filter_timing=None, loglikelihood_burn=None):
        if filter_method is None:
            filter_method = self.filter_method
        if inversion_method is None:
            inversion_method = self.inversion_method
        if stability_method is None:
            stability_method = self.stability_method
        if conserve_memory is None:
            conserve_memory = self.conserve_memory
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        if filter_timing is None:
            filter_timing = self.filter_timing
        if tolerance is None:
            tolerance = self.tolerance
        if self.endog is None:
            raise RuntimeError('Must bind a dataset to the model before filtering or smoothing.')
        prefix, dtype, create_statespace = self._initialize_representation()
        create_filter = create_statespace or prefix not in self._kalman_filters
        if not create_filter:
            kalman_filter = self._kalman_filters[prefix]
            create_filter = not kalman_filter.conserve_memory == conserve_memory or not kalman_filter.loglikelihood_burn == loglikelihood_burn
        if create_filter:
            if prefix in self._kalman_filters:
                del self._kalman_filters[prefix]
            cls = self.prefix_kalman_filter_map[prefix]
            self._kalman_filters[prefix] = cls(self._statespaces[prefix], filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn)
        else:
            kalman_filter = self._kalman_filters[prefix]
            kalman_filter.set_filter_method(filter_method, False)
            kalman_filter.inversion_method = inversion_method
            kalman_filter.stability_method = stability_method
            kalman_filter.filter_timing = filter_timing
            kalman_filter.tolerance = tolerance
        return (prefix, dtype, create_filter, create_statespace)

    def set_filter_method(self, filter_method=None, **kwargs):
        """
        Set the filtering method

        The filtering method controls aspects of which Kalman filtering
        approach will be used.

        Parameters
        ----------
        filter_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The filtering method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        FILTER_CONVENTIONAL
            Conventional Kalman filter.
        FILTER_UNIVARIATE
            Univariate approach to Kalman filtering. Overrides conventional
            method if both are specified.
        FILTER_COLLAPSED
            Collapsed approach to Kalman filtering. Will be used *in addition*
            to conventional or univariate filtering.
        FILTER_CONCENTRATED
            Use the concentrated log-likelihood function. Will be used
            *in addition* to the other options.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `filter_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is FILTER_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.filter_method
        1
        >>> mod.ssm.filter_conventional
        True
        >>> mod.ssm.filter_univariate = True
        >>> mod.ssm.filter_method
        17
        >>> mod.ssm.set_filter_method(filter_univariate=False,
        ...                           filter_collapsed=True)
        >>> mod.ssm.filter_method
        33
        >>> mod.ssm.set_filter_method(filter_method=1)
        >>> mod.ssm.filter_conventional
        True
        >>> mod.ssm.filter_univariate
        False
        >>> mod.ssm.filter_collapsed
        False
        >>> mod.ssm.filter_univariate = True
        >>> mod.ssm.filter_method
        17
        """
        if filter_method is not None:
            self.filter_method = filter_method
        for name in KalmanFilter.filter_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_inversion_method(self, inversion_method=None, **kwargs):
        """
        Set the inversion method

        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.

        Parameters
        ----------
        inversion_method : int, optional
            Bitmask value to set the inversion method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the inversion method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The inversion method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        INVERT_UNIVARIATE
            If the endogenous time series is univariate, then inversion can be
            performed by simple division. If this flag is set and the time
            series is univariate, then division will always be used even if
            other flags are also set.
        SOLVE_LU
            Use an LU decomposition along with a linear solver (rather than
            ever actually inverting the matrix).
        INVERT_LU
            Use an LU decomposition along with typical matrix inversion.
        SOLVE_CHOLESKY
            Use a Cholesky decomposition along with a linear solver.
        INVERT_CHOLESKY
            Use an Cholesky decomposition along with typical matrix inversion.

        If the bitmask is set directly via the `inversion_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the inversion method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default inversion method is `INVERT_UNIVARIATE | SOLVE_CHOLESKY`

        Several things to keep in mind are:

        - If the filtering method is specified to be univariate, then simple
          division is always used regardless of the dimension of the endogenous
          time series.
        - Cholesky decomposition is about twice as fast as LU decomposition,
          but it requires that the matrix be positive definite. While this
          should generally be true, it may not be in every case.
        - Using a linear solver rather than true matrix inversion is generally
          faster and is numerically more stable.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.inversion_method
        1
        >>> mod.ssm.solve_cholesky
        True
        >>> mod.ssm.invert_univariate
        True
        >>> mod.ssm.invert_lu
        False
        >>> mod.ssm.invert_univariate = False
        >>> mod.ssm.inversion_method
        8
        >>> mod.ssm.set_inversion_method(solve_cholesky=False,
        ...                              invert_cholesky=True)
        >>> mod.ssm.inversion_method
        16
        """
        if inversion_method is not None:
            self.inversion_method = inversion_method
        for name in KalmanFilter.inversion_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_stability_method(self, stability_method=None, **kwargs):
        """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : int, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The stability method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        STABILITY_FORCE_SYMMETRY = 0x01
            If this flag is set, symmetry of the predicted state covariance
            matrix is enforced at each iteration of the filter, where each
            element is set to the average of the corresponding elements in the
            upper and lower triangle.

        If the bitmask is set directly via the `stability_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the stability method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default stability method is `STABILITY_FORCE_SYMMETRY`

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.stability_method
        1
        >>> mod.ssm.stability_force_symmetry
        True
        >>> mod.ssm.stability_force_symmetry = False
        >>> mod.ssm.stability_method
        0
        """
        if stability_method is not None:
            self.stability_method = stability_method
        for name in KalmanFilter.stability_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : int, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.

        Notes
        -----
        The memory conservation method is defined by a collection of boolean
        flags, and is internally stored as a bitmask. The methods available
        are:

        MEMORY_STORE_ALL
            Store all intermediate matrices. This is the default value.
        MEMORY_NO_FORECAST_MEAN
            Do not store the forecast or forecast errors. If this option is
            used, the `predict` method from the results class is unavailable.
        MEMORY_NO_FORECAST_COV
            Do not store the forecast error covariance matrices.
        MEMORY_NO_FORECAST
            Do not store the forecast, forecast error, or forecast error
            covariance matrices. If this option is used, the `predict` method
            from the results class is unavailable.
        MEMORY_NO_PREDICTED_MEAN
            Do not store the predicted state.
        MEMORY_NO_PREDICTED_COV
            Do not store the predicted state covariance
            matrices.
        MEMORY_NO_PREDICTED
            Do not store the predicted state or predicted state covariance
            matrices.
        MEMORY_NO_FILTERED_MEAN
            Do not store the filtered state.
        MEMORY_NO_FILTERED_COV
            Do not store the filtered state covariance
            matrices.
        MEMORY_NO_FILTERED
            Do not store the filtered state or filtered state covariance
            matrices.
        MEMORY_NO_LIKELIHOOD
            Do not store the vector of loglikelihood values for each
            observation. Only the sum of the loglikelihood values is stored.
        MEMORY_NO_GAIN
            Do not store the Kalman gain matrices.
        MEMORY_NO_SMOOTHING
            Do not store temporary variables related to Kalman smoothing. If
            this option is used, smoothing is unavailable.
        MEMORY_NO_STD_FORECAST
            Do not store standardized forecast errors.
        MEMORY_CONSERVE
            Do not store any intermediate matrices.

        If the bitmask is set directly via the `conserve_memory` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the memory conservation method may also be specified by
        directly modifying the class attributes which are defined similarly to
        the keyword arguments.

        The default memory conservation method is `MEMORY_STORE_ALL`, so that
        all intermediate matrices are stored.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm..conserve_memory
        0
        >>> mod.ssm.memory_no_predicted
        False
        >>> mod.ssm.memory_no_predicted = True
        >>> mod.ssm.conserve_memory
        2
        >>> mod.ssm.set_conserve_memory(memory_no_filtered=True,
        ...                             memory_no_forecast=True)
        >>> mod.ssm.conserve_memory
        7
        """
        if conserve_memory is not None:
            self.conserve_memory = conserve_memory
        for name in KalmanFilter.memory_options:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_filter_timing(self, alternate_timing=None, **kwargs):
        """
        Set the filter timing convention

        By default, the Kalman filter follows Durbin and Koopman, 2012, in
        initializing the filter with predicted values. Kim and Nelson, 1999,
        instead initialize the filter with filtered values, which is
        essentially just a different timing convention.

        Parameters
        ----------
        alternate_timing : int, optional
            Whether or not to use the alternate timing convention. Default is
            unspecified.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.
        """
        if alternate_timing is not None:
            self.filter_timing = int(alternate_timing)
        if 'timing_init_predicted' in kwargs:
            self.filter_timing = int(not kwargs['timing_init_predicted'])
        if 'timing_init_filtered' in kwargs:
            self.filter_timing = int(kwargs['timing_init_filtered'])

    @contextlib.contextmanager
    def fixed_scale(self, scale):
        """
        fixed_scale(scale)

        Context manager for fixing the scale when FILTER_CONCENTRATED is set

        Parameters
        ----------
        scale : numeric
            Scale of the model.

        Notes
        -----
        This a no-op if scale is None.

        This context manager is most useful in models which are explicitly
        concentrating out the scale, so that the set of parameters they are
        estimating does not include the scale.
        """
        if scale is not None and scale != 1:
            if not self.filter_concentrated:
                raise ValueError('Cannot provide scale if filter method does not include FILTER_CONCENTRATED.')
            self.filter_concentrated = False
            self._scale = scale
            obs_cov = self['obs_cov']
            state_cov = self['state_cov']
            self['obs_cov'] = scale * obs_cov
            self['state_cov'] = scale * state_cov
        try:
            yield
        finally:
            if scale is not None and scale != 1:
                self['state_cov'] = state_cov
                self['obs_cov'] = obs_cov
                self.filter_concentrated = True
                self._scale = None

    def _filter(self, filter_method=None, inversion_method=None, stability_method=None, conserve_memory=None, filter_timing=None, tolerance=None, loglikelihood_burn=None, complex_step=False):
        prefix, dtype, create_filter, create_statespace = self._initialize_filter(filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn)
        kfilter = self._kalman_filters[prefix]
        self._initialize_state(prefix=prefix, complex_step=complex_step)
        kfilter()
        return kfilter

    def filter(self, filter_method=None, inversion_method=None, stability_method=None, conserve_memory=None, filter_timing=None, tolerance=None, loglikelihood_burn=None, complex_step=False):
        """
        Apply the Kalman filter to the statespace model.

        Parameters
        ----------
        filter_method : int, optional
            Determines which Kalman filter to use. Default is conventional.
        inversion_method : int, optional
            Determines which inversion technique to use. Default is by Cholesky
            decomposition.
        stability_method : int, optional
            Determines which numerical stability techniques to use. Default is
            to enforce symmetry of the predicted state covariance matrix.
        conserve_memory : int, optional
            Determines what output from the filter to store. Default is to
            store everything.
        filter_timing : int, optional
            Determines the timing convention of the filter. Default is that
            from Durbin and Koopman (2012), in which the filter is initialized
            with predicted values.
        tolerance : float, optional
            The tolerance at which the Kalman filter determines convergence to
            steady-state. Default is 1e-19.
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.

        Notes
        -----
        This function by default does not compute variables required for
        smoothing.
        """
        if conserve_memory is None:
            conserve_memory = self.conserve_memory | MEMORY_NO_SMOOTHING
        conserve_memory_cache = self.conserve_memory
        self.set_conserve_memory(conserve_memory)
        kfilter = self._filter(filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn, complex_step)
        results = self.results_class(self)
        results.update_representation(self)
        results.update_filter(kfilter)
        self.set_conserve_memory(conserve_memory_cache)
        return results

    def loglike(self, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """
        kwargs.setdefault('conserve_memory', MEMORY_CONSERVE ^ MEMORY_NO_LIKELIHOOD)
        kfilter = self._filter(**kwargs)
        loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
        if not kwargs['conserve_memory'] & MEMORY_NO_LIKELIHOOD:
            loglike = np.sum(kfilter.loglikelihood[loglikelihood_burn:])
        else:
            loglike = np.sum(kfilter.loglikelihood)
        if self.filter_method & FILTER_CONCENTRATED:
            d = max(loglikelihood_burn, kfilter.nobs_diffuse)
            nobs_k_endog = np.sum(self.k_endog - np.array(self._statespace.nmissing)[d:])
            nobs_k_endog -= kfilter.nobs_kendog_univariate_singular
            if not kwargs['conserve_memory'] & MEMORY_NO_LIKELIHOOD:
                scale = np.sum(kfilter.scale[d:]) / nobs_k_endog
            else:
                scale = kfilter.scale[0] / nobs_k_endog
            loglike += -0.5 * nobs_k_endog
            if kfilter.nobs_diffuse > 0:
                nobs_k_endog -= kfilter.nobs_kendog_diffuse_nonsingular
            loglike += -0.5 * nobs_k_endog * np.log(scale)
        return loglike

    def loglikeobs(self, **kwargs):
        """
        Calculate the loglikelihood for each observation associated with the
        statespace model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        If `loglikelihood_burn` is positive, then the entries in the returned
        loglikelihood vector are set to be zero for those initial time periods.

        Returns
        -------
        loglike : array of float
            Array of loglikelihood values for each observation.
        """
        if self.memory_no_likelihood:
            raise RuntimeError('Cannot compute loglikelihood if MEMORY_NO_LIKELIHOOD option is selected.')
        if not self.filter_method & FILTER_CONCENTRATED:
            kwargs.setdefault('conserve_memory', MEMORY_CONSERVE ^ MEMORY_NO_LIKELIHOOD)
        else:
            kwargs.setdefault('conserve_memory', MEMORY_CONSERVE ^ (MEMORY_NO_FORECAST | MEMORY_NO_LIKELIHOOD))
        kfilter = self._filter(**kwargs)
        llf_obs = np.array(kfilter.loglikelihood, copy=True)
        loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
        if self.filter_method & FILTER_CONCENTRATED:
            d = max(loglikelihood_burn, kfilter.nobs_diffuse)
            nmissing = np.array(self._statespace.nmissing)
            nobs_k_endog = np.sum(self.k_endog - nmissing[d:])
            nobs_k_endog -= kfilter.nobs_kendog_univariate_singular
            scale = np.sum(kfilter.scale[d:]) / nobs_k_endog
            nsingular = 0
            if kfilter.nobs_diffuse > 0:
                d = kfilter.nobs_diffuse
                Finf = kfilter.forecast_error_diffuse_cov
                singular = np.diagonal(Finf).real <= kfilter.tolerance_diffuse
                nsingular = np.sum(~singular, axis=1)
            scale_obs = np.array(kfilter.scale, copy=True)
            llf_obs += -0.5 * ((self.k_endog - nmissing - nsingular) * np.log(scale) + scale_obs / scale)
        llf_obs[:loglikelihood_burn] = 0
        return llf_obs

    def simulate(self, nsimulations, measurement_shocks=None, state_shocks=None, initial_state=None, pretransformed_measurement_shocks=True, pretransformed_state_shocks=True, pretransformed_initial_state=True, simulator=None, return_simulator=False, random_state=None):
        """
        Simulate a new time series following the state space model

        Parameters
        ----------
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used. If unspecified and
            the model has not been initialized, then a vector of zeros is used.
            Note that this is not included in the returned `simulated_states`
            array.
        pretransformed_measurement_shocks : bool, optional
            If `measurement_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `obs_cov` covariance matrix. Default is True.
        pretransformed_state_shocks : bool, optional
            If `state_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `state_cov` covariance matrix. Default is
            True.
        pretransformed_initial_state : bool, optional
            If `initial_state` is provided, this flag indicates whether it
            should be directly used as the initial_state. If False, then it is
            assumed to contain draws from the standard Normal distribution that
            must be transformed using the `initial_state_cov` covariance
            matrix. Default is True.
        return_simulator : bool, optional
            Whether or not to return the simulator object. Typically used to
            improve performance when performing repeated sampling. Default is
            False.
        random_state : {None, int, Generator, RandomState}, optionall
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        simulated_obs : ndarray
            An (nsimulations x k_endog) array of simulated observations.
        simulated_states : ndarray
            An (nsimulations x k_states) array of simulated states.
        simulator : SimulationSmoothResults
            If `return_simulator=True`, then an instance of a simulator is
            returned, which can be reused for additional simulations of the
            same size.
        """
        time_invariant = self.time_invariant
        if not time_invariant and nsimulations > self.nobs:
            raise ValueError('In a time-varying model, cannot create more simulations than there are observations.')
        return self._simulate(nsimulations, measurement_disturbance_variates=measurement_shocks, state_disturbance_variates=state_shocks, initial_state_variates=initial_state, pretransformed_measurement_disturbance_variates=pretransformed_measurement_shocks, pretransformed_state_disturbance_variates=pretransformed_state_shocks, pretransformed_initial_state_variates=pretransformed_initial_state, simulator=simulator, return_simulator=return_simulator, random_state=random_state)

    def _simulate(self, nsimulations, simulator=None, random_state=None, **kwargs):
        raise NotImplementedError('Simulation only available through the simulation smoother.')

    def impulse_responses(self, steps=10, impulse=0, orthogonalized=False, cumulative=False, direct=False):
        """
        Impulse response function

        Parameters
        ----------
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 10. Note that the initial impulse is not counted as a
            step, so if `steps=1`, the output will have 2 entries.
        impulse : int or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1` where `k_posdef` is the same as in the state
            space model. Alternatively, a custom impulse vector may be
            provided; must be a column vector with shape `(k_posdef, 1)`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. A (steps + 1 x k_endog) array.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        TODO: add note about how for time-varying systems this is - perhaps
        counter-intuitively - returning the impulse response within the given
        model (i.e. starting at period 0 defined by the model) and it is *not*
        doing impulse responses after the end of the model. To compute impulse
        responses from arbitrary time points, it is necessary to clone a new
        model with the appropriate system matrices.
        """
        steps += 1
        if self._design.shape[2] == 1 and self._transition.shape[2] == 1 and (self._selection.shape[2] == 1):
            steps += 1
        if type(impulse) is int:
            if impulse >= self.k_posdef or impulse < 0:
                raise ValueError('Invalid value for `impulse`. Must be the index of one of the state innovations.')
            idx = impulse
            impulse = np.zeros(self.k_posdef)
            impulse[idx] = 1
        else:
            impulse = np.array(impulse)
            if impulse.ndim > 1:
                impulse = np.squeeze(impulse)
            if not impulse.shape == (self.k_posdef,):
                raise ValueError('Invalid impulse vector. Must be shaped (%d,)' % self.k_posdef)
        if orthogonalized:
            state_chol = np.linalg.cholesky(self.state_cov[:, :, 0])
            impulse = np.dot(state_chol, impulse)
        time_invariant_irf = self._design.shape[2] == self._transition.shape[2] == self._selection.shape[2] == 1
        if not time_invariant_irf and steps > self.nobs:
            raise ValueError('In a time-varying model, cannot create more impulse responses than there are observations')
        sim_model = self.clone(endog=np.zeros((steps, self.k_endog), dtype=self.dtype), obs_intercept=np.zeros(self.k_endog), design=self['design', :, :, :steps], obs_cov=np.zeros((self.k_endog, self.k_endog)), state_intercept=np.zeros(self.k_states), transition=self['transition', :, :, :steps], selection=self['selection', :, :, :steps], state_cov=np.zeros((self.k_posdef, self.k_posdef)))
        measurement_shocks = np.zeros((steps, self.k_endog))
        state_shocks = np.zeros((steps, self.k_posdef))
        state_shocks[0] = impulse
        initial_state = np.zeros((self.k_states,))
        irf, _ = sim_model.simulate(steps, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        if cumulative:
            irf = np.cumsum(irf, axis=0)
        return irf[1:]