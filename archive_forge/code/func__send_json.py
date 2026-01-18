import json
import time
import uuid
from typing import cast, Dict, List, Sequence, Tuple, Union
import numpy as np
from requests import put
import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string
def _send_json(self, *, json_str: str, id_str: Union[str, uuid.UUID], repetitions: int=1, num_qubits: int=1) -> np.ndarray:
    """Replaces the remote host with a local simulator

        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an ndarray of booleans.
        """
    sim = AQTSimulator(num_qubits=num_qubits, simulate_ideal=self.simulate_ideal)
    sim.generate_circuit_from_list(json_str)
    data = sim.simulate_samples(repetitions)
    return data.measurements['m']