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
def _make_label(ae: str, value: Any) -> str | None:
    if not isinstance(value, stage):
        return _nice_label(value)
    elif value.start is None:
        if value.after_stat is not None:
            return value.after_stat
        elif value.after_scale is not None:
            return value.after_scale
        else:
            raise ValueError('Unknown mapping')
    elif value.after_stat is not None:
        return value.after_stat
    else:
        return _nice_label(value)