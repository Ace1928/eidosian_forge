import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
class KrausModel(_KrausModel):
    """
    Encapsulate a single gate's noise model.

    :ivar str gate: The name of the gate.
    :ivar Sequence[float] params: Optional parameters for the gate.
    :ivar Sequence[int] targets: The target qubit ids.
    :ivar Sequence[np.array] kraus_ops: The Kraus operators (must be square complex numpy arrays).
    :ivar float fidelity: The average gate fidelity associated with the Kraus map relative to the
        ideal operation.
    """

    @staticmethod
    def unpack_kraus_matrix(m: Union[List[Any], np.ndarray]) -> np.ndarray:
        """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param m: The representation of a Kraus operator. Either a complex
            square matrix (as numpy array or nested lists) or a JSON-able pair of real matrices
            (as nested lists) representing the element-wise real and imaginary part of m.
        :return: A complex square numpy array representing the Kraus operator.
        """
        m = np.asarray(m, dtype=complex)
        if m.ndim == 3:
            m = m[0] + 1j * m[1]
        if not m.ndim == 2:
            raise ValueError('Need 2d array.')
        if not m.shape[0] == m.shape[1]:
            raise ValueError('Need square matrix.')
        return m

    def to_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary representation of a KrausModel.

        For example::

            {
                "gate": "RX",
                "params": np.pi,
                "targets": [0],
                "kraus_ops": [            # In this example single Kraus op = ideal RX(pi) gate
                    [[[0,   0],           # element-wise real part of matrix
                      [0,   0]],
                      [[0, -1],           # element-wise imaginary part of matrix
                      [-1, 0]]]
                ],
                "fidelity": 1.0
            }

        :return: A JSON compatible dictionary representation.
        :rtype: Dict[str,Any]
        """
        res = self._asdict()
        res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
        return res

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'KrausModel':
        """
        Recreate a KrausModel from the dictionary representation.

        :param d: The dictionary representing the KrausModel. See `to_dict` for an
            example.
        :return: The deserialized KrausModel.
        """
        kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
        return KrausModel(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other: object) -> bool:
        return not self.__eq__(other)