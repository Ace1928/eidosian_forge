import numpy as np
from onnx.reference.op_run import OpRun
class NegativeLogLikelihoodLoss(OpRun):

    def _run(self, x, target, weight=None, ignore_index=None, reduction=None):
        return _compute_negative_log_likelihood_loss(x, target, weight=weight, reduction=reduction, ignore_index=ignore_index)