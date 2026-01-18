from typing import Sequence, TYPE_CHECKING
from cirq import circuits, devices, value, ops
from cirq.devices.noise_model import validate_all_measurements
class DampedReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with T1 decay preceding measurement.

    This simulates asymmetric readout error. Note that since noise is applied
    before the measurement moment, composing this model on top of another noise
    model will place the T1 decay immediately before the measurement
    (regardless of the previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, decay_prob: float, prepend: bool=True):
        """A depolarizing noise model with damped readout error.

        Args:
            decay_prob: Probability of T1 decay during measurement.
            prepend: If True, put noise before affected gates. Default: True.
        """
        value.validate_probability(decay_prob, 'decay_prob')
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)
        self._prepend = prepend

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if self.is_virtual_moment(moment):
            return moment
        if validate_all_measurements(moment):
            output = [circuits.Moment((self.readout_decay_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits)), moment]
            return output if self._prepend else output[::-1]
        return moment