from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _apply_kraus(kraus: Union[Tuple[np.ndarray], Sequence[Any]], args: 'ApplyChannelArgs') -> np.ndarray:
    """Directly apply the kraus operators to the target tensor."""
    args.out_buffer[:] = 0
    np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)
    if len(args.left_axes) == 1 and kraus[0].shape == (2, 2):
        return _apply_kraus_single_qubit(kraus, args)
    return _apply_kraus_multi_qubit(kraus, args)