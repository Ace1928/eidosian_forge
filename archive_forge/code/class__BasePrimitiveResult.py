from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Dict
from numpy import ndarray
class _BasePrimitiveResult(ABC):
    """
    Base class for deprecated Primitive result methods.
    """

    def __post_init__(self) -> None:
        """
        Verify that all fields in any inheriting result dataclass are consistent, after
        instantiation, with the number of experiments being represented.

        This magic method is specific of `dataclasses.dataclass`, therefore all inheriting
        classes must have this decorator.

        Raises:
            TypeError: If one of the data fields is not a Sequence or ``numpy.ndarray``.
            ValueError: Inconsistent number of experiments across data fields.
        """
        num_experiments = None
        for value in self._field_values:
            if num_experiments is None:
                num_experiments = len(value)
            if not isinstance(value, (Sequence, ndarray)) or isinstance(value, (str, bytes)):
                raise TypeError(f'Expected sequence or `numpy.ndarray`, provided {type(value)} instead.')
            if len(value) != num_experiments:
                raise ValueError('Inconsistent number of experiments across data fields.')

    @property
    def _field_names(self) -> tuple[str, ...]:
        """Tuple of field names in any inheriting result dataclass."""
        return tuple((field.name for field in fields(self)))

    @property
    def _field_values(self) -> tuple:
        """Tuple of field values in any inheriting result dataclass."""
        return tuple((getattr(self, name) for name in self._field_names))