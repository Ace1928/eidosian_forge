import numpy as np
from onnx.reference.op_run import OpRun
class SoftmaxCrossEntropyLoss(OpRun):

    def _run(self, x, target, weight=None, ignore_index=None, reduction=None):
        n_outputs = len(self.onnx_node.output)
        return softmaxcrossentropy(x, target, weight=weight, reduction=reduction, ignore_index=ignore_index, get_log_prob=n_outputs == 2)