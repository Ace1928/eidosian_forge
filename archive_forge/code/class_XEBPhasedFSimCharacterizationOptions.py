import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
@dataclasses.dataclass(frozen=True)
class XEBPhasedFSimCharacterizationOptions(XEBCharacterizationOptions):
    """Options for calibrating a PhasedFSim-like gate using XEB.

    You may want to use more specific subclasses like `SqrtISwapXEBOptions`
    which have sensible defaults.

    Attributes:
        characterize_theta: Whether to characterize θ angle.
        characterize_zeta: Whether to characterize ζ angle.
        characterize_chi: Whether to characterize χ angle.
        characterize_gamma: Whether to characterize γ angle.
        characterize_phi: Whether to characterize φ angle.
        theta_default: The initial or default value to assume for the θ angle.
        zeta_default: The initial or default value to assume for the ζ angle.
        chi_default: The initial or default value to assume for the χ angle.
        gamma_default: The initial or default value to assume for the γ angle.
        phi_default: The initial or default value to assume for the φ angle.
    """
    characterize_theta: bool = True
    characterize_zeta: bool = True
    characterize_chi: bool = True
    characterize_gamma: bool = True
    characterize_phi: bool = True
    theta_default: Optional[float] = None
    zeta_default: Optional[float] = None
    chi_default: Optional[float] = None
    gamma_default: Optional[float] = None
    phi_default: Optional[float] = None

    def _iter_angles(self) -> Iterable[Tuple[bool, Optional[float], 'sympy.Symbol']]:
        yield from ((self.characterize_theta, self.theta_default, THETA_SYMBOL), (self.characterize_zeta, self.zeta_default, ZETA_SYMBOL), (self.characterize_chi, self.chi_default, CHI_SYMBOL), (self.characterize_gamma, self.gamma_default, GAMMA_SYMBOL), (self.characterize_phi, self.phi_default, PHI_SYMBOL))

    def _iter_angles_for_characterization(self) -> Iterable[Tuple[Optional[float], 'sympy.Symbol']]:
        yield from ((default, symbol) for characterize, default, symbol in self._iter_angles() if characterize)

    def get_initial_simplex_and_names(self, initial_simplex_step_size: float=0.1) -> Tuple[np.ndarray, List[str]]:
        """Get an initial simplex and parameter names for the optimization implied by these options.

        The initial simplex initiates the Nelder-Mead optimization parameter. We
        use the standard simplex of `x0 + s*basis_vec` where x0 is given by the
        `xxx_default` attributes, s is `initial_simplex_step_size` and `basis_vec`
        is a one-hot encoded vector for each parameter for which the `parameterize_xxx`
        attribute is True.

        We also return a list of parameter names so the Cirq `param_resovler`
        can be accurately constructed during optimization.
        """
        x0_list = []
        names = []
        for default, symbol in self._iter_angles_for_characterization():
            if default is None:
                raise ValueError(f'{symbol.name}_default was not set.')
            x0_list.append(default)
            names.append(symbol.name)
        x0 = np.asarray(x0_list)
        n_param = len(x0)
        initial_simplex = [x0]
        for i in range(n_param):
            basis_vec = np.eye(1, n_param, i)[0]
            initial_simplex += [x0 + initial_simplex_step_size * basis_vec]
        return (np.asarray(initial_simplex), names)

    def get_parameterized_gate(self):
        theta = THETA_SYMBOL if self.characterize_theta else self.theta_default
        zeta = ZETA_SYMBOL if self.characterize_zeta else self.zeta_default
        chi = CHI_SYMBOL if self.characterize_chi else self.chi_default
        gamma = GAMMA_SYMBOL if self.characterize_gamma else self.gamma_default
        phi = PHI_SYMBOL if self.characterize_phi else self.phi_default
        return ops.PhasedFSimGate(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi)

    @staticmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        return isinstance(op.gate, (ops.PhasedFSimGate, ops.ISwapPowGate, ops.FSimGate))

    def defaults_set(self) -> bool:
        """Whether the default angles are set.

        This only considers angles where characterize_{angle} is True. If all such angles have
        {angle}_default set to a value, this returns True. If none of the defaults are set,
        this returns False. If some defaults are set, we raise an exception.
        """
        defaults_set = [default is not None for _, default, _ in self._iter_angles()]
        if any(defaults_set):
            if all(defaults_set):
                return True
            problems = [symbol.name for _, default, symbol in self._iter_angles() if default is None]
            raise ValueError(f'Some angles are set, but values for {problems} are not.')
        return False

    def with_defaults_from_gate(self, gate: 'cirq.Gate', gate_to_angles_func=phased_fsim_angles_from_gate):
        """A new Options class with {angle}_defaults inferred from `gate`.

        This keeps the same settings for the characterize_{angle} booleans, but will disregard
        any current {angle}_default values.
        """
        return XEBPhasedFSimCharacterizationOptions(characterize_theta=self.characterize_theta, characterize_zeta=self.characterize_zeta, characterize_chi=self.characterize_chi, characterize_gamma=self.characterize_gamma, characterize_phi=self.characterize_phi, **gate_to_angles_func(gate))

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)