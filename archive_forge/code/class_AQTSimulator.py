import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
class AQTSimulator:
    """A simulator for the AQT device."""

    def __init__(self, num_qubits: int, circuit: cirq.Circuit=cirq.Circuit(), simulate_ideal: bool=False, noise_dict: Optional[Dict]=None):
        """Initializes the AQT simulator.

        Args:
            num_qubits: Number of qubits.
            circuit: Optional, circuit to be simulated.
                Last moment needs to be a measurement over all qubits with key 'm'
            simulate_ideal: If True, an ideal, noiseless, circuit will be simulated.
            noise_dict: A map from gate to noise to be applied after that gate. If None, uses
                a default noise model.
        """
        self.circuit = circuit
        self.num_qubits = num_qubits
        self.qubit_list = cirq.LineQubit.range(num_qubits)
        if noise_dict is None:
            noise_dict = get_default_noise_dict()
        self.noise_dict = noise_dict
        self.simulate_ideal = simulate_ideal

    def generate_circuit_from_list(self, json_string: str):
        """Generates a list of cirq operations from a json string.

        The default behavior is to add a measurement to any qubit at the end
        of the circuit as there are no measurements defined in the AQT API.

        Args:
            json_string: json that specifies the sequence.
        """
        self.circuit = cirq.Circuit()
        json_obj = json.loads(json_string)
        gate: Union[cirq.PhasedXPowGate, cirq.EigenGate]
        for circuit_list in json_obj:
            op_str = circuit_list[0]
            if op_str == 'R':
                gate = cast(cirq.PhasedXPowGate, gate_dict[op_str])
                theta = circuit_list[1]
                phi = circuit_list[2]
                qubits = [self.qubit_list[i] for i in circuit_list[3]]
                self.circuit.append(gate(phase_exponent=phi, exponent=theta).on(*qubits))
            else:
                gate = cast(cirq.EigenGate, gate_dict[op_str])
                angle = circuit_list[1]
                qubits = [self.qubit_list[i] for i in circuit_list[2]]
                self.circuit.append(gate.on(*qubits) ** angle)
        self.circuit.append(cirq.measure(*[qubit for qubit in self.qubit_list], key='m'))

    def simulate_samples(self, repetitions: int) -> cirq.Result:
        """Samples the circuit.

        Args:
            repetitions: Number of times the circuit is simulated.

        Returns:
            Result from Cirq.Simulator.

        Raises:
            RuntimeError: Simulate called without a circuit.
        """
        if self.simulate_ideal:
            noise_model = cirq.NO_NOISE
        else:
            noise_model = AQTNoiseModel()
        if self.circuit == cirq.Circuit():
            raise RuntimeError('Simulate called without a valid circuit.')
        sim = cirq.DensityMatrixSimulator(noise=noise_model)
        result = sim.run(self.circuit, repetitions=repetitions)
        return result