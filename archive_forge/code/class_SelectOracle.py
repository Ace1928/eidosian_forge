import abc
from typing import Tuple
from cirq._compat import cached_property
from cirq_ft import infra
class SelectOracle(infra.GateWithRegisters):
    """Abstract base class that defines the API for a SELECT Oracle.

    The action of a SELECT oracle on a selection register $|l\\rangle$ and target register
    $|\\Psi\\rangle$ can be defined as:

    $$
        \\mathrm{SELECT} = \\sum_{l}|l \\rangle \\langle l| \\otimes U_l
    $$

    In other words, the `SELECT` oracle applies $l$'th unitary $U_{l}$ on the target register
    $|\\Psi\\rangle$ when the selection register stores integer $l$.

    $$
        \\mathrm{SELECT}|l\\rangle |\\Psi\\rangle = |l\\rangle U_{l}|\\Psi\\rangle
    $$
    """

    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[infra.Register, ...]:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[infra.Register, ...]:
        ...

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers, *self.target_registers])