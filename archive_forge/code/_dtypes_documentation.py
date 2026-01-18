import datetime
import math
import typing as t
from wandb.util import (
Explain why an item is not assignable to a type.

        Assumes that the caller has already validated that the assignment fails.

        Args:
            other (any): Any object depth (int, optional): depth of the type checking.
                Defaults to 0.

        Returns:
            str: human-readable explanation
        