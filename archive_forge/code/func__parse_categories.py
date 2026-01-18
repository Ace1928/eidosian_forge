from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _parse_categories(things: Collection[int], stuffs: Collection[int]) -> Tuple[Set[int], Set[int]]:
    """Parse and validate metrics arguments for `things` and `stuff`.

    Args:
        things: All possible IDs for things categories.
        stuffs: All possible IDs for stuff categories.

    Returns:
        things_parsed: A set of unique category IDs for the things categories.
        stuffs_parsed: A set of unique category IDs for the stuffs categories.

    """
    things_parsed = set(things)
    if len(things_parsed) < len(things):
        rank_zero_warn('The provided `things` categories contained duplicates, which have been removed.', UserWarning)
    stuffs_parsed = set(stuffs)
    if len(stuffs_parsed) < len(stuffs):
        rank_zero_warn('The provided `stuffs` categories contained duplicates, which have been removed.', UserWarning)
    if not all((isinstance(val, int) for val in things_parsed)):
        raise TypeError(f'Expected argument `things` to contain `int` categories, but got {things}')
    if not all((isinstance(val, int) for val in stuffs_parsed)):
        raise TypeError(f'Expected argument `stuffs` to contain `int` categories, but got {stuffs}')
    if things_parsed & stuffs_parsed:
        raise ValueError(f'Expected arguments `things` and `stuffs` to have distinct keys, but got {things} and {stuffs}')
    if not things_parsed | stuffs_parsed:
        raise ValueError('At least one of `things` and `stuffs` must be non-empty.')
    return (things_parsed, stuffs_parsed)