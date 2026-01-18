import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
@staticmethod
def _post_process_predicted_label(label, scores, classlabels_ints_string):
    """Replaces int64 predicted labels by the corresponding
        strings.
        """
    if classlabels_ints_string is not None:
        label = np.array([classlabels_ints_string[i] for i in label])
    return (label, scores)