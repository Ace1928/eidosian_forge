from typing import List, Tuple
import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_properties import NoiseProperties, NoiseModelFromNoiseProperties
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
class SampleNoiseProperties(NoiseProperties):

    def __init__(self, system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]):
        self.qubits = system_qubits
        self.qubit_pairs = qubit_pairs

    def build_noise_models(self):
        add_h = InsertionNoiseModel({OpIdentifier(cirq.Gate, q): cirq.H(q) for q in self.qubits})
        add_iswap = InsertionNoiseModel({OpIdentifier(cirq.Gate, *qs): cirq.ISWAP(*qs) for qs in self.qubit_pairs})
        return [add_h, add_iswap]