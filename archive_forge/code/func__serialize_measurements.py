import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_measurements(self, meas_ops: Iterator) -> Dict[str, str]:
    """Serializes measurement ops into a form suitable to be passed via metadata.

        IonQ API does not contain measurement gates, so we serialize measurement gate keys
        and targets into a form that is suitable for passing through IonQ's metadata field
        for a job.

        Each key and targets are serialized into a string of the form `key` + the ASCII unit
        separator (chr(31)) + targets as a comma separated value.  These are then combined
        into a string with a separator character of the ASCII record separator (chr(30)).
        Finally this full string is serialized as the values in the metadata dict with keys
        given by `measurementX` for X = 0,1, .. 9 and X large enough to contain the entire
        string.

        Args:
            A list of the result of serializing the measurement (not supported by the API).

        Returns:
            The metadata dict that can be passed to the API.

        Raises:
            ValueError: if the
        """
    key_values = [f'{op['key']}{chr(31)}{op['targets']}' for op in meas_ops]
    full_str = chr(30).join(key_values)
    max_value_size = 40
    split_strs = [full_str[i:i + max_value_size] for i in range(0, len(full_str), max_value_size)]
    if len(split_strs) > 9:
        raise ValueError('Measurement keys plus target strings too long for IonQ API. Please use smaller keys.')
    return {f'measurement{i}': x for i, x in enumerate(split_strs)}