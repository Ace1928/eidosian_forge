from typing import Optional, Union, Tuple
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.exceptions import PulseError
class ShiftFrequency(Instruction):
    """Shift the channel frequency away from the current frequency."""

    def __init__(self, frequency: Union[float, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        """Creates a new shift frequency instruction.

        Args:
            frequency: Frequency shift of the channel in Hz.
            channel: The channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        super().__init__(operands=(frequency, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """Frequency shift from the set frequency."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0