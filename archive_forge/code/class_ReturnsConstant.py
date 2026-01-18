import numpy as np
import cirq
class ReturnsConstant:

    def __init__(self, bound):
        self.bound = bound

    def _trace_distance_bound_(self) -> float:
        return self.bound