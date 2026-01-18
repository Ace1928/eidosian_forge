import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_regular_values_close(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool, identifier: Optional[Union[str, Callable[[str], str]]]=None) -> None:
    """Checks if the values of two tensors are close up to a desired tolerance."""
    matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if torch.all(matches):
        return
    if actual.shape == torch.Size([]):
        msg = make_scalar_mismatch_msg(actual.item(), expected.item(), rtol=rtol, atol=atol, identifier=identifier)
    else:
        msg = make_tensor_mismatch_msg(actual, expected, matches, rtol=rtol, atol=atol, identifier=identifier)
    self._fail(AssertionError, msg)