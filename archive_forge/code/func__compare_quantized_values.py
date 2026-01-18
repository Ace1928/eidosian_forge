import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_quantized_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
    """Compares quantized tensors by comparing the :meth:`~torch.Tensor.dequantize`'d variants for closeness.

        .. note::

            A detailed discussion about why only the dequantized variant is checked for closeness rather than checking
            the individual quantization parameters for closeness and the integer representation for equality can be
            found in https://github.com/pytorch/pytorch/issues/68548.
        """
    return self._compare_regular_values_close(actual.dequantize(), expected.dequantize(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier=lambda default_identifier: f'Quantized {default_identifier.lower()}')