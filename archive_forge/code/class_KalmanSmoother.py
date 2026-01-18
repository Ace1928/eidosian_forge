import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
class KalmanSmoother(KalmanFilter):
    """
    State space representation of a time series process, with Kalman filter
    and smoother.

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `SmootherResults`. If specified, class must extend from
        `SmootherResults`.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, or for Kalman smoothing
        options. See `Representation` for more details.
    """
    smoother_outputs = ['smoother_state', 'smoother_state_cov', 'smoother_state_autocov', 'smoother_disturbance', 'smoother_disturbance_cov', 'smoother_all']
    smoother_state = OptionWrapper('smoother_output', SMOOTHER_STATE)
    smoother_state_cov = OptionWrapper('smoother_output', SMOOTHER_STATE_COV)
    smoother_disturbance = OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE)
    smoother_disturbance_cov = OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE_COV)
    smoother_state_autocov = OptionWrapper('smoother_output', SMOOTHER_STATE_AUTOCOV)
    smoother_all = OptionWrapper('smoother_output', SMOOTHER_ALL)
    smooth_methods = ['smooth_conventional', 'smooth_alternative', 'smooth_classical']
    smooth_conventional = OptionWrapper('smooth_method', SMOOTH_CONVENTIONAL)
    '\n    (bool) Flag for conventional (Durbin and Koopman, 2012) Kalman smoothing.\n    '
    smooth_alternative = OptionWrapper('smooth_method', SMOOTH_ALTERNATIVE)
    '\n    (bool) Flag for alternative (modified Bryson-Frazier) smoothing.\n    '
    smooth_classical = OptionWrapper('smooth_method', SMOOTH_CLASSICAL)
    '\n    (bool) Flag for classical (see e.g. Anderson and Moore, 1979) smoothing.\n    '
    smooth_univariate = OptionWrapper('smooth_method', SMOOTH_UNIVARIATE)
    '\n    (bool) Flag for univariate smoothing (uses modified Bryson-Frazier timing).\n    '
    smoother_output = SMOOTHER_ALL
    smooth_method = 0

    def __init__(self, k_endog, k_states, k_posdef=None, results_class=None, kalman_smoother_classes=None, **kwargs):
        if results_class is None:
            results_class = SmootherResults
        keys = ['smoother_output'] + KalmanSmoother.smoother_outputs
        smoother_output_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['smooth_method'] + KalmanSmoother.smooth_methods
        smooth_method_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        super().__init__(k_endog, k_states, k_posdef, results_class=results_class, **kwargs)
        self.prefix_kalman_smoother_map = kalman_smoother_classes if kalman_smoother_classes is not None else tools.prefix_kalman_smoother_map.copy()
        self._kalman_smoothers = {}
        self.set_smoother_output(**smoother_output_kwargs)
        self.set_smooth_method(**smooth_method_kwargs)

    def _clone_kwargs(self, endog, **kwargs):
        kwargs = super()._clone_kwargs(endog, **kwargs)
        kwargs.setdefault('smoother_output', self.smoother_output)
        kwargs.setdefault('smooth_method', self.smooth_method)
        return kwargs

    @property
    def _kalman_smoother(self):
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    def _initialize_smoother(self, smoother_output=None, smooth_method=None, prefix=None, **kwargs):
        if smoother_output is None:
            smoother_output = self.smoother_output
        if smooth_method is None:
            smooth_method = self.smooth_method
        prefix, dtype, create_filter, create_statespace = self._initialize_filter(prefix, **kwargs)
        create_smoother = create_filter or prefix not in self._kalman_smoothers
        if not create_smoother:
            kalman_smoother = self._kalman_smoothers[prefix]
            create_smoother = kalman_smoother.kfilter is not self._kalman_filters[prefix]
        if create_smoother:
            cls = self.prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(self._statespaces[prefix], self._kalman_filters[prefix], smoother_output, smooth_method)
        else:
            self._kalman_smoothers[prefix].set_smoother_output(smoother_output, False)
            self._kalman_smoothers[prefix].set_smooth_method(smooth_method)
        return (prefix, dtype, create_smoother, create_filter, create_statespace)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        """
        Set the smoother output

        The smoother can produce several types of results. The smoother output
        variable controls which are calculated and returned.

        Parameters
        ----------
        smoother_output : int, optional
            Bitmask value to set the smoother output to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the smoother output by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoother output is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTHER_STATE = 0x01
            Calculate and return the smoothed states.
        SMOOTHER_STATE_COV = 0x02
            Calculate and return the smoothed state covariance matrices.
        SMOOTHER_STATE_AUTOCOV = 0x10
            Calculate and return the smoothed state lag-one autocovariance
            matrices.
        SMOOTHER_DISTURBANCE = 0x04
            Calculate and return the smoothed state and observation
            disturbances.
        SMOOTHER_DISTURBANCE_COV = 0x08
            Calculate and return the covariance matrices for the smoothed state
            and observation disturbances.
        SMOOTHER_ALL
            Calculate and return all results.

        If the bitmask is set directly via the `smoother_output` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the smoother output may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default smoother output is SMOOTHER_ALL.

        If performance is a concern, only those results which are needed should
        be specified as any results that are not specified will not be
        calculated. For example, if the smoother output is set to only include
        SMOOTHER_STATE, the smoother operates much more quickly than if all
        output is required.

        Examples
        --------
        >>> import statsmodels.tsa.statespace.kalman_smoother as ks
        >>> mod = ks.KalmanSmoother(1,1)
        >>> mod.smoother_output
        15
        >>> mod.set_smoother_output(smoother_output=0)
        >>> mod.smoother_state = True
        >>> mod.smoother_output
        1
        >>> mod.smoother_state
        True
        """
        if smoother_output is not None:
            self.smoother_output = smoother_output
        for name in KalmanSmoother.smoother_outputs:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_smooth_method(self, smooth_method=None, **kwargs):
        """
        Set the smoothing method

        The smoothing method can be used to override the Kalman smoother
        approach used. By default, the Kalman smoother used depends on the
        Kalman filter method.

        Parameters
        ----------
        smooth_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoothing method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTH_CONVENTIONAL = 0x01
            Default Kalman smoother, as presented in Durbin and Koopman, 2012
            chapter 4.
        SMOOTH_CLASSICAL = 0x02
            Classical Kalman smoother, as presented in Anderson and Moore, 1979
            or Durbin and Koopman, 2012 chapter 4.6.1.
        SMOOTH_ALTERNATIVE = 0x04
            Modified Bryson-Frazier Kalman smoother method; this is identical
            to the conventional method of Durbin and Koopman, 2012, except that
            an additional intermediate step is included.
        SMOOTH_UNIVARIATE = 0x08
            Univariate Kalman smoother, as presented in Durbin and Koopman,
            2012 chapter 6, except with modified Bryson-Frazier timing.

        Practically speaking, these methods should all produce the same output
        but different computational implications, numerical stability
        implications, or internal timing assumptions.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `smooth_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is SMOOTH_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.smooth_method
        1
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        >>> mod.set_smooth_method(filter_univariate=False,
                                  filter_collapsed=True)
        >>> mod.smooth_method
        33
        >>> mod.set_smooth_method(smooth_method=1)
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate
        False
        >>> mod.filter_collapsed
        False
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        """
        if smooth_method is not None:
            self.smooth_method = smooth_method
        for name in KalmanSmoother.smooth_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def _smooth(self, smoother_output=None, smooth_method=None, prefix=None, complex_step=False, results=None, **kwargs):
        prefix, dtype, create_smoother, create_filter, create_statespace = self._initialize_smoother(smoother_output, smooth_method, prefix=prefix, **kwargs)
        if create_filter or create_statespace:
            raise ValueError('Passed settings forced re-creation of the Kalman filter. Please run `_filter` before running `_smooth`.')
        smoother = self._kalman_smoothers[prefix]
        smoother()
        return smoother

    def smooth(self, smoother_output=None, smooth_method=None, results=None, run_filter=True, prefix=None, complex_step=False, update_representation=True, update_filter=True, update_smoother=True, **kwargs):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of both filtering and smoothing.
            If an object, then that object is updated with the smoothing data.
            If None, then a SmootherResults object is returned with both
            filtering and smoothing results.
        run_filter : bool, optional
            Whether or not to run the Kalman filter prior to smoothing. Default
            is True.
        prefix : str
            The prefix of the datatype. Usually only used internally.

        Returns
        -------
        SmootherResults object
        """
        kfilter = self._filter(**kwargs)
        results = self.results_class(self)
        if update_representation:
            results.update_representation(self)
        if update_filter:
            results.update_filter(kfilter)
        else:
            results.nobs_diffuse = kfilter.nobs_diffuse
        if smoother_output is None:
            smoother_output = self.smoother_output
        smoother = self._smooth(smoother_output, results=results, **kwargs)
        if update_smoother:
            results.update_smoother(smoother)
        return results