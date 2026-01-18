import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def _probabilities(self, scores, class_count_):
    probsp2 = np.zeros((class_count_, class_count_), dtype=scores.dtype)
    index = 0
    for i in range(class_count_):
        for j in range(i + 1, class_count_):
            val1 = sigmoid_probability(scores[index], self._svm.atts.prob_a[index], self._svm.atts.prob_b[index])
            val2 = max(val1, 1e-07)
            val2 = min(val2, 1 - 1e-07)
            probsp2[i, j] = val2
            probsp2[j, i] = 1 - val2
            index += 1
    return multiclass_probability(class_count_, probsp2)