import numpy as np
from onnx.reference.op_run import OpRun
class TopK_1(_CommonTopK):

    def _run(self, data, k=None, axis=None):
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
        return _CommonTopK._common_run(self, data, [k], axis=axis)