from typing import Optional, Tuple
import numpy as np
from numpy.random import RandomState  # type: ignore
from onnx.reference.op_run import OpRun
def _private_run(self, X: np.ndarray, seed: Optional[int]=None, ratio: float=0.5, training_mode: bool=False) -> Tuple[np.ndarray]:
    return _dropout(X, ratio, seed=seed, return_mask=self.n_outputs == 2, training_mode=training_mode)