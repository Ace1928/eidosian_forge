import pickle
import sys
import time
import torch
import torch.utils.benchmark as benchmark_utils
class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """

    def __init__(self, real_torch, extra_ns_per_element):
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    def extra_overhead(self, result):
        numel = int(result.numel())
        if numel > 5000:
            time.sleep(numel * self._extra_ns_per_element * 1e-09)
        return result

    def add(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.add(*args, **kwargs))

    def mul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

    def cat(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

    def matmul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))