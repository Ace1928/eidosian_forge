import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
class FixedQParamsFakeQuantize(FakeQuantize):
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    def __init__(self, observer):
        super().__init__(observer=observer)
        assert type(self.activation_post_process) == FixedQParamsObserver, f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
        self._observer_ctr = observer
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        assert _is_per_tensor(self.qscheme), 'Only per tensor quantization is supported' + ' FixedQParamsFakeQuantize module, got qscheme:' + str(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return (self.scale, self.zero_point)

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        return 'fake_quant_enabled={}, observer_enabled={}, scale={}, zero_point={}, dtype={}, quant_min={}, quant_max={}, qscheme={}'.format(self.fake_quant_enabled, self.observer_enabled, self.scale, self.zero_point, self.dtype, self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.qscheme)