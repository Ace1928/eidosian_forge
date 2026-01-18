from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
@property
def _starting(self) -> dict[str, Any]:
    """
        Return the subset of aesthetics mapped from the layer data

        The mapping is a dict of the form ``{name: expr}``, i.e the
        stage class has been peeled off.
        """
    d = {}
    for name, value in self.items():
        if not isinstance(value, stage):
            d[name] = value
        elif isinstance(value, stage) and value.start is not None:
            d[name] = value.start
    return d