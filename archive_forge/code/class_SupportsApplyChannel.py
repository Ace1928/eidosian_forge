from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
class SupportsApplyChannel(Protocol):
    """An object that can efficiently implement a channel."""

    @doc_private
    def _apply_channel_(self, args: ApplyChannelArgs) -> Union[np.ndarray, None, NotImplementedType]:
        """Efficiently applies a channel.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        a workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        Args:
            args: A `cirq.ApplyChannelArgs` object with the `args.target_tensor`
                to operate on, an `args.out_buffer`, 'args.auxiliary_buffer0`
                and `args.auxiliary_buffer1` buffers to use as temporary
                workspace, and the `args.left_axes` and `args.right_axes` of
                the tensor to target with the unitary operation. Note that
                this method is permitted (and in fact expected) to mutate
                `args.target_tensor` and the given buffers.

        Returns:
            If the receiving object is not able to apply a channel, None
            or NotImplemented should be returned.

            If the receiving object is able to work inline, it should directly
            mutate `args.target_tensor` and then return `args.target_tensor`.
            The caller will understand this to mean that the result is in
            `args.target_tensor`.

            If the receiving object is unable to work inline, it can write its
            output over `args.out_buffer` and then return `args.out_buffer`.
            The caller will understand this to mean that the result is in
            `args.out_buffer` (and so what was `args.out` will become
            `args.target_tensor` in the next call, and vice versa).

            The receiving object is also permitted to allocate a new
            numpy.ndarray and return that as its result.
        """