from math import exp
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import cirq
from cirq.devices.noise_model import validate_all_measurements
class PerQubitDepolarizingWithDampedReadoutNoiseModel(cirq.NoiseModel):
    """NoiseModel with T1 decay on gates and damping/bitflip on measurement.

    With this model, T1 decay is added after all non-measurement gates, then
    amplitude damping followed by bitflip error is added before all measurement
    gates. Idle qubits are unaffected by this model.

    As with the DepolarizingWithDampedReadoutNoiseModel, this model does not
    allow a moment to contain both measurement and non-measurement gates.
    """

    def __init__(self, depol_probs: Optional[Dict[cirq.Qid, float]]=None, bitflip_probs: Optional[Dict[cirq.Qid, float]]=None, decay_probs: Optional[Dict[cirq.Qid, float]]=None):
        """A depolarizing noise model with damped readout error.

        All error modes are specified on a per-qubit basis. To omit a given
        error mode from the noise model, leave its dict blank when initializing
        this object.

        Args:
            depol_probs: Dict of depolarizing probabilities for each qubit.
            bitflip_probs: Dict of bit-flip probabilities during measurement.
            decay_probs: Dict of T1 decay probabilities during measurement.
                Bitflip noise is applied first, then amplitude decay.
        """
        for probs, desc in [(depol_probs, 'depolarization prob'), (bitflip_probs, 'readout error prob'), (decay_probs, 'readout decay prob')]:
            if probs:
                for qubit, prob in probs.items():
                    cirq.validate_probability(prob, f'{desc} of {qubit}')
        self.depol_probs = depol_probs
        self.bitflip_probs = bitflip_probs
        self.decay_probs = decay_probs

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self.is_virtual_moment(moment):
            return moment
        moments = []
        if validate_all_measurements(moment):
            if self.decay_probs:
                moments.append(cirq.Moment((cirq.AmplitudeDampingChannel(self.decay_probs[q])(q) for q in system_qubits)))
            if self.bitflip_probs:
                moments.append(cirq.Moment((cirq.BitFlipChannel(self.bitflip_probs[q])(q) for q in system_qubits)))
            moments.append(moment)
            return moments
        else:
            moments.append(moment)
            if self.depol_probs:
                gated_qubits = [q for q in system_qubits if moment.operates_on_single_qubit(q)]
                if gated_qubits:
                    moments.append(cirq.Moment((cirq.DepolarizingChannel(self.depol_probs[q])(q) for q in gated_qubits)))
            return moments