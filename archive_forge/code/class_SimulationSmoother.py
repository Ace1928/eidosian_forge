import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
class SimulationSmoother(KalmanSmoother):
    """
    State space representation of a time series process, with Kalman filter
    and smoother, and with simulation smoother.

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
    simulation_smooth_results_class : class, optional
        Default results class to use to save output of simulation smoothing.
        Default is `SimulationSmoothResults`. If specified, class must extend
        from `SimulationSmoothResults`.
    simulation_smoother_classes : dict, optional
        Dictionary with BLAS prefixes as keys and classes as values.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, for Kalman smoothing
        options, or for Simulation smoothing options.
        See `Representation`, `KalmanFilter`, and `KalmanSmoother` for more
        details.
    """
    simulation_outputs = ['simulate_state', 'simulate_disturbance', 'simulate_all']

    def __init__(self, k_endog, k_states, k_posdef=None, simulation_smooth_results_class=None, simulation_smoother_classes=None, **kwargs):
        super().__init__(k_endog, k_states, k_posdef, **kwargs)
        if simulation_smooth_results_class is None:
            simulation_smooth_results_class = SimulationSmoothResults
        self.simulation_smooth_results_class = simulation_smooth_results_class
        self.prefix_simulation_smoother_map = simulation_smoother_classes if simulation_smoother_classes is not None else tools.prefix_simulation_smoother_map.copy()
        self._simulators = {}

    def get_simulation_output(self, simulation_output=None, simulate_state=None, simulate_disturbance=None, simulate_all=None, **kwargs):
        """
        Get simulation output bitmask

        Helper method to get final simulation output bitmask from a set of
        optional arguments including the bitmask itself and possibly boolean
        flags.

        Parameters
        ----------
        simulation_output : int, optional
            Simulation output bitmask. If this is specified, it is simply
            returned and the other arguments are ignored.
        simulate_state : bool, optional
            Whether or not to include the state in the simulation output.
        simulate_disturbance : bool, optional
            Whether or not to include the state and observation disturbances
            in the simulation output.
        simulate_all : bool, optional
            Whether or not to include all simulation output.
        \\*\\*kwargs
            Additional keyword arguments. Present so that calls to this method
            can use \\*\\*kwargs without clearing out additional arguments.
        """
        if simulation_output is None:
            simulation_output = 0
            if simulate_state:
                simulation_output |= SIMULATION_STATE
            if simulate_disturbance:
                simulation_output |= SIMULATION_DISTURBANCE
            if simulate_all:
                simulation_output |= SIMULATION_ALL
            if simulation_output == 0:
                argument_set = not all([simulate_state is None, simulate_disturbance is None, simulate_all is None])
                if argument_set:
                    raise ValueError('Invalid simulation output options: given options would result in no output.')
                simulation_output = self.smoother_output
        return simulation_output

    def _simulate(self, nsimulations, simulator=None, random_state=None, return_simulator=False, **kwargs):
        if simulator is None:
            simulator = self.simulator(nsimulations, random_state=random_state)
        simulator.simulate(**kwargs)
        simulated_obs = np.array(simulator.generated_obs, copy=True)
        simulated_state = np.array(simulator.generated_state, copy=True)
        out = (simulated_obs.T[:nsimulations], simulated_state.T[:nsimulations])
        if return_simulator:
            out = out + (simulator,)
        return out

    def simulator(self, nsimulations, random_state=None):
        return self.simulation_smoother(simulation_output=0, method='kfs', nobs=nsimulations, random_state=random_state)

    def simulation_smoother(self, simulation_output=None, method='kfs', results_class=None, prefix=None, nobs=-1, random_state=None, **kwargs):
        """
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        method : {'kfs', 'cfa'}, optional
            Method for simulation smoothing. If `method='kfs'`, then the
            simulation smoother is based on Kalman filtering and smoothing
            recursions. If `method='cfa'`, then the simulation smoother is
            based on the Cholesky Factor Algorithm (CFA) approach. The CFA
            approach is not applicable to all state space models, but can be
            faster for the cases in which it is supported.
        results_class : class, optional
            Default results class to use to save output of simulation
            smoothing. Default is `SimulationSmoothResults`. If specified,
            class must extend from `SimulationSmoothResults`.
        prefix : str
            The prefix of the datatype. Usually only used internally.
        nobs : int
            The number of observations to simulate. If set to anything other
            than -1, only simulation will be performed (i.e. simulation
            smoothing will not be performed), so that only the `generated_obs`
            and `generated_state` attributes will be available.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``numpy.random.RandomState`` instance
            is used, seeded with `seed`.
            If `seed` is already a ``numpy.random.Generator`` or
            ``numpy.random.RandomState`` instance then that instance is used.
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        method = method.lower()
        if method == 'cfa':
            if simulation_output not in [None, 1, -1]:
                raise ValueError('Can only retrieve simulations of the state vector using the CFA simulation smoother.')
            return CFASimulationSmoother(self)
        elif method != 'kfs':
            raise ValueError('Invalid simulation smoother method "%s". Valid methods are "kfs" or "cfa".' % method)
        if results_class is None:
            results_class = self.simulation_smooth_results_class
        if not issubclass(results_class, SimulationSmoothResults):
            raise ValueError('Invalid results class provided.')
        prefix, dtype, create_smoother, create_filter, create_statespace = self._initialize_smoother()
        simulation_output = self.get_simulation_output(simulation_output, **kwargs)
        smoother_output = kwargs.get('smoother_output', simulation_output)
        filter_method = kwargs.get('filter_method', self.filter_method)
        inversion_method = kwargs.get('inversion_method', self.inversion_method)
        stability_method = kwargs.get('stability_method', self.stability_method)
        conserve_memory = kwargs.get('conserve_memory', self.conserve_memory)
        filter_timing = kwargs.get('filter_timing', self.filter_timing)
        loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
        tolerance = kwargs.get('tolerance', self.tolerance)
        cls = self.prefix_simulation_smoother_map[prefix]
        simulation_smoother = cls(self._statespaces[prefix], filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn, smoother_output, simulation_output, nobs)
        results = results_class(self, simulation_smoother, random_state=random_state)
        return results