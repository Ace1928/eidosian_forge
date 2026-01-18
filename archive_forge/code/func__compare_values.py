import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_values(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.is_quantized:
        compare_fn = self._compare_quantized_values
    elif actual.is_sparse:
        compare_fn = self._compare_sparse_coo_values
    elif actual.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        compare_fn = self._compare_sparse_compressed_values
    else:
        compare_fn = self._compare_regular_values_close
    compare_fn(actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan)