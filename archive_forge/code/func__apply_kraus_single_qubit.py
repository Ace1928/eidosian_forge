from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _apply_kraus_single_qubit(kraus: Union[Tuple[Any], Sequence[Any]], args: 'ApplyChannelArgs') -> np.ndarray:
    """Use slicing to apply single qubit channel.  Only for two-level qubits."""
    zero_left = linalg.slice_for_qubits_equal_to(args.left_axes, 0)
    one_left = linalg.slice_for_qubits_equal_to(args.left_axes, 1)
    zero_right = linalg.slice_for_qubits_equal_to(args.right_axes, 0)
    one_right = linalg.slice_for_qubits_equal_to(args.right_axes, 1)
    for kraus_op in kraus:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
        linalg.apply_matrix_to_slices(args.target_tensor, kraus_op, [zero_left, one_left], out=args.auxiliary_buffer1)
        linalg.apply_matrix_to_slices(args.auxiliary_buffer1, np.conjugate(kraus_op), [zero_right, one_right], out=args.target_tensor)
        args.out_buffer += args.target_tensor
    return args.out_buffer