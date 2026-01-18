import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def _validate_pruning_amount_init(amount):
    """Validate helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(f'Invalid type for amount: {amount}. Must be int or float.')
    if isinstance(amount, numbers.Integral) and amount < 0 or (not isinstance(amount, numbers.Integral) and (float(amount) > 1.0 or float(amount) < 0.0)):
        raise ValueError(f'amount={amount} should either be a float in the range [0, 1] or a non-negative integer')