import numpy as np
from onnx.reference.op_run import OpRun
class _CommonTopK(OpRun):

    def _common_run(self, data, ink, axis, largest=1):
        """Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what `onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        k = ink[0]
        axis = axis if axis >= 0 else axis + len(data.shape)
        sort, sorti = topk_sorted_implementation(data, k, axis, largest)
        return (sort, sorti.astype(np.int64))