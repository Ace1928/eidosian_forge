import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _make_mismatch_msg(*, default_identifier: str, identifier: Optional[Union[str, Callable[[str], str]]]=None, extra: Optional[str]=None, abs_diff: float, abs_diff_idx: Optional[Union[int, Tuple[int, ...]]]=None, atol: float, rel_diff: float, rel_diff_idx: Optional[Union[int, Tuple[int, ...]]]=None, rtol: float) -> str:
    """Makes a mismatch error message for numeric values.

    Args:
        default_identifier (str): Default description of the compared values, e.g. "Tensor-likes".
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier that overrides
            ``default_identifier``. Can be passed as callable in which case it will be called with
            ``default_identifier`` to create the description at runtime.
        extra (Optional[str]): Extra information to be placed after the message header and the mismatch statistics.
        abs_diff (float): Absolute difference.
        abs_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the absolute difference.
        atol (float): Allowed absolute tolerance. Will only be added to mismatch statistics if it or ``rtol`` are
            ``> 0``.
        rel_diff (float): Relative difference.
        rel_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the relative difference.
        rtol (float): Allowed relative tolerance. Will only be added to mismatch statistics if it or ``atol`` are
            ``> 0``.
    """
    equality = rtol == 0 and atol == 0

    def make_diff_msg(*, type: str, diff: float, idx: Optional[Union[int, Tuple[int, ...]]], tol: float) -> str:
        if idx is None:
            msg = f'{type.title()} difference: {diff}'
        else:
            msg = f'Greatest {type} difference: {diff} at index {idx}'
        if not equality:
            msg += f' (up to {tol} allowed)'
        return msg + '\n'
    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)
    msg = f'{identifier} are not {('equal' if equality else 'close')}!\n\n'
    if extra:
        msg += f'{extra.strip()}\n'
    msg += make_diff_msg(type='absolute', diff=abs_diff, idx=abs_diff_idx, tol=atol)
    msg += make_diff_msg(type='relative', diff=rel_diff, idx=rel_diff_idx, tol=rtol)
    return msg.strip()