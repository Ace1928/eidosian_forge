import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
class DefaultGaussian(Device):
    """Default Gaussian device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. If ``None``, the results are analytically computed and hence deterministic.
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`
    """
    name = 'Default Gaussian PennyLane plugin'
    short_name = 'default.gaussian'
    pennylane_requires = __version__
    version = __version__
    author = 'Xanadu Inc.'
    _operation_map = {'Identity': Identity.identity_op, 'Snapshot': None, 'Beamsplitter': beamsplitter, 'ControlledAddition': controlled_addition, 'ControlledPhase': controlled_phase, 'Displacement': displacement, 'QuadraticPhase': quadratic_phase, 'Rotation': rotation, 'Squeezing': squeezing, 'TwoModeSqueezing': two_mode_squeezing, 'CoherentState': coherent_state, 'DisplacedSqueezedState': displaced_squeezed_state, 'SqueezedState': squeezed_state, 'ThermalState': thermal_state, 'GaussianState': gaussian_state, 'InterferometerUnitary': interferometer_unitary}
    _observable_map = {'NumberOperator': photon_number, 'QuadX': homodyne(0), 'QuadP': homodyne(np.pi / 2), 'QuadOperator': homodyne(None), 'PolyXP': poly_quad_expectations, 'FockStateProjector': fock_expectation, 'Identity': identity}
    _circuits = {}

    def __init__(self, wires, *, shots=None, hbar=2, analytic=None):
        super().__init__(wires, shots, analytic=analytic)
        self.eng = None
        self.hbar = hbar
        self._debugger = None
        self.reset()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(model='cv', supports_analytic_computation=True, supports_finite_shots=True, returns_probs=False, returns_state=False)
        return capabilities

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        device_wires = self.map_wires(wires)
        if operation == 'Displacement':
            self._state = displacement(self._state, device_wires.labels[0], par[0] * cmath.exp(1j * par[1]))
            return
        if operation == 'GaussianState':
            if len(device_wires) != self.num_wires:
                raise ValueError('GaussianState covariance matrix or means vector is the incorrect size for the number of subsystems.')
            self._state = self._operation_map[operation](*par, hbar=self.hbar)
            return
        if operation == 'Snapshot':
            if self._debugger and self._debugger.active:
                gaussian = {'cov_matrix': self._state[0].copy(), 'means': self._state[1].copy()}
                self._debugger.snapshots[len(self._debugger.snapshots)] = gaussian
            return
        if 'State' in operation:
            cov, mu = self._operation_map[operation](*par, hbar=self.hbar)
            self._state = set_state(self._state, device_wires[:1], cov, mu)
            return
        S = self._operation_map[operation](*par)
        S = self.expand(S, device_wires)
        means = S @ self._state[1]
        cov = S @ self._state[0] @ S.T
        self._state = [cov, means]

    def expand(self, S, wires):
        """Expands a Symplectic matrix S to act on the entire subsystem.

        Args:
            S (array): a :math:`2M\\times 2M` Symplectic matrix
            wires (Wires): wires of the modes that S acts on

        Returns:
            array: the resulting :math:`2N\\times 2N` Symplectic matrix
        """
        if self.num_wires == 1:
            return S
        N = self.num_wires
        w = wires.toarray()
        M = len(S) // 2
        S2 = np.identity(2 * N)
        if M != len(wires):
            raise ValueError('Incorrect number of subsystems for provided operation.')
        S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:M, :M].copy()
        S2[(w + N).reshape(-1, 1), (w + N).reshape(1, -1)] = S[M:, M:].copy()
        S2[w.reshape(-1, 1), (w + N).reshape(1, -1)] = S[:M, M:].copy()
        S2[(w + N).reshape(-1, 1), w.reshape(1, -1)] = S[M:, :M].copy()
        return S2

    def expval(self, observable, wires, par):
        if observable == 'PolyXP':
            cov, mu = self._state
            ev, var = self._observable_map[observable](cov, mu, wires, self.wires, par, hbar=self.hbar)
        else:
            cov, mu = self.reduced_state(wires)
            ev, var = self._observable_map[observable](cov, mu, par, hbar=self.hbar)
        if self.shots is not None:
            ev = np.random.normal(ev, math.sqrt(var / self.shots))
        return ev

    def var(self, observable, wires, par):
        if observable == 'PolyXP':
            cov, mu = self._state
            _, var = self._observable_map[observable](cov, mu, wires, self.wires, par, hbar=self.hbar)
        else:
            cov, mu = self.reduced_state(wires)
            _, var = self._observable_map[observable](cov, mu, par, hbar=self.hbar)
        return var

    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        .. note::

            The ``default.gaussian`` plugin only supports sampling
            from :class:`~.X`, :class:`~.P`, and :class:`~.QuadOperator`
            observables.

        Args:
            observable (str): name of the observable
            wires (Wires): wires the observable is to be measured on
            par (tuple): parameters for the observable

        Returns:
            array[float]: samples in an array of dimension ``(n, num_wires)``
        """
        if len(wires) != 1:
            raise ValueError('Only one mode can be measured in homodyne.')
        if observable == 'QuadX':
            phi = 0.0
        elif observable == 'QuadP':
            phi = np.pi / 2
        elif observable == 'QuadOperator':
            phi = par[0]
        else:
            raise NotImplementedError(f'default.gaussian does not support sampling {observable}')
        cov, mu = self.reduced_state(wires)
        rot = rotation(phi)
        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot
        stdphi = math.sqrt(covphi[0, 0])
        meanphi = muphi[0]
        return np.random.normal(meanphi, stdphi, self.shots)

    def reset(self):
        """Reset the device"""
        self._state = vacuum_state(self.num_wires, self.hbar)

    def reduced_state(self, wires):
        """Returns the covariance matrix and the vector of means of the specified wires.

        Args:
            wires (Wires): requested wires

        Returns:
            tuple (cov, means): cov is a square array containing the covariance matrix,
            and means is an array containing the vector of means
        """
        if len(wires) == self.num_wires:
            return self._state
        device_wires = self.map_wires(wires)
        ind = np.concatenate([device_wires.toarray(), device_wires.toarray() + self.num_wires])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)
        return (self._state[0][rows, cols], self._state[1][ind])

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())

    def execute(self, operations, observables):
        if len(observables) > 1:
            raise qml.QuantumFunctionError('Default gaussian only support single measurements.')
        return super().execute(operations, observables)

    def batch_execute(self, circuits):
        results = super().batch_execute(circuits)
        results = [qml.math.squeeze(res) for res in results]
        return results