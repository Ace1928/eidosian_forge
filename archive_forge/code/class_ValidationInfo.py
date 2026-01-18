from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class ValidationInfo(Protocol):
    """
    Argument passed to validation functions.
    """

    @property
    def context(self) -> Any | None:
        """Current validation context."""
        ...

    @property
    def config(self) -> CoreConfig | None:
        """The CoreConfig that applies to this validation."""
        ...

    @property
    def mode(self) -> Literal['python', 'json']:
        """The type of input data we are currently validating"""
        ...

    @property
    def data(self) -> Dict[str, Any]:
        """The data being validated for this model."""
        ...

    @property
    def field_name(self) -> str | None:
        """
        The name of the current field being validated if this validator is
        attached to a model field.
        """
        ...