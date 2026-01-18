from typing import Sequence, TYPE_CHECKING
from cirq import circuits, devices, value, ops
from cirq.devices.noise_model import validate_all_measurements
class DepolarizingWithReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingNoiseModel with probabilistic bit flips preceding
    measurement.
    This simulates readout error.
    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float):
        """A depolarizing noise model with readout error.
        Args:
            depol_prob: Depolarizing probability.
            bitflip_prob: Probability of a bit-flip during measurement.
        """
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if validate_all_measurements(moment):
            return [circuits.Moment((self.readout_noise_gate(q) for q in system_qubits)), moment]
        return [moment, circuits.Moment((self.qubit_noise_gate(q) for q in system_qubits))]