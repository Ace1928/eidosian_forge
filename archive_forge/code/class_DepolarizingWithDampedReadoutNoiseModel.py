from typing import Sequence, TYPE_CHECKING
from cirq import circuits, devices, value, ops
from cirq.devices.noise_model import validate_all_measurements
class DepolarizingWithDampedReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingWithReadoutNoiseModel with T1 decay preceding
    measurement.
    This simulates asymmetric readout error. The noise is structured
    so the T1 decay is applied, then the readout bitflip, then measurement.
    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float, decay_prob: float):
        """A depolarizing noise model with damped readout error.
        Args:
            depol_prob: Depolarizing probability.
            bitflip_prob: Probability of a bit-flip during measurement.
            decay_prob: Probability of T1 decay during measurement.
                Bitflip noise is applied first, then amplitude decay.
        """
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        value.validate_probability(decay_prob, 'decay_prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if validate_all_measurements(moment):
            return [circuits.Moment((self.readout_decay_gate(q) for q in system_qubits)), circuits.Moment((self.readout_noise_gate(q) for q in system_qubits)), moment]
        else:
            return [moment, circuits.Moment((self.qubit_noise_gate(q) for q in system_qubits))]