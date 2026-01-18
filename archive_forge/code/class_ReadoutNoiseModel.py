from typing import Sequence, TYPE_CHECKING
from cirq import circuits, devices, value, ops
from cirq.devices.noise_model import validate_all_measurements
class ReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with probabilistic bit flips preceding measurement.

    This simulates readout error. Note that since noise is applied before the
    measurement moment, composing this model on top of another noise model will
    place the bit flips immediately before the measurement (regardless of the
    previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, bitflip_prob: float, prepend: bool=True):
        """A noise model with readout error.

        Args:
            bitflip_prob: Probability of a bit-flip during measurement.
            prepend: If True, put noise before affected gates. Default: True.
        """
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)
        self._prepend = prepend

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if self.is_virtual_moment(moment):
            return moment
        if validate_all_measurements(moment):
            output = [circuits.Moment((self.readout_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits)), moment]
            return output if self._prepend else output[::-1]
        return moment