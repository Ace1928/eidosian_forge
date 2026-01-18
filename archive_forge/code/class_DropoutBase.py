from typing import Optional, Tuple
import numpy as np
from numpy.random import RandomState  # type: ignore
from onnx.reference.op_run import OpRun
class DropoutBase(OpRun):

    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def _private_run(self, X: np.ndarray, seed: Optional[int]=None, ratio: float=0.5, training_mode: bool=False) -> Tuple[np.ndarray]:
        return _dropout(X, ratio, seed=seed, return_mask=self.n_outputs == 2, training_mode=training_mode)