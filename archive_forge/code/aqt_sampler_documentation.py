import json
import time
import uuid
from typing import cast, Dict, List, Sequence, Tuple, Union
import numpy as np
from requests import put
import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string
Replaces the remote host with a local simulator

        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an ndarray of booleans.
        