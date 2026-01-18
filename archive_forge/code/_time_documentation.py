import math as _math
import time as _time
import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.api import runtime
A :class:`numpy.ndarray` of shape ``(len(devices), n_repeat)``,
        holding times spent on GPU in seconds.

        These values are measured using ``cudaEventElapsedTime`` with events
        recoreded before/after each repeat step.
        