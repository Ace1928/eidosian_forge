import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_attributes(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Checks if the attributes of two tensors match.

        Always checks

        - the :attr:`~torch.Tensor.shape`,
        - whether both inputs are quantized or not,
        - and if they use the same quantization scheme.

        Checks for

        - :attr:`~torch.Tensor.layout`,
        - :meth:`~torch.Tensor.stride`,
        - :attr:`~torch.Tensor.device`, and
        - :attr:`~torch.Tensor.dtype`

        are optional and can be disabled through the corresponding ``check_*`` flag during construction of the pair.
        """

    def raise_mismatch_error(attribute_name: str, actual_value: Any, expected_value: Any) -> NoReturn:
        self._fail(AssertionError, f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.")
    if actual.shape != expected.shape:
        raise_mismatch_error('shape', actual.shape, expected.shape)
    if actual.is_quantized != expected.is_quantized:
        raise_mismatch_error('is_quantized', actual.is_quantized, expected.is_quantized)
    elif actual.is_quantized and actual.qscheme() != expected.qscheme():
        raise_mismatch_error('qscheme()', actual.qscheme(), expected.qscheme())
    if actual.layout != expected.layout:
        if self.check_layout:
            raise_mismatch_error('layout', actual.layout, expected.layout)
    elif actual.layout == torch.strided and self.check_stride and (actual.stride() != expected.stride()):
        raise_mismatch_error('stride()', actual.stride(), expected.stride())
    if self.check_device and actual.device != expected.device:
        raise_mismatch_error('device', actual.device, expected.device)
    if self.check_dtype and actual.dtype != expected.dtype:
        raise_mismatch_error('dtype', actual.dtype, expected.dtype)