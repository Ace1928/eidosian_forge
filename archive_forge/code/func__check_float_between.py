from __future__ import annotations
import math
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Progress_pb2 import Progress as ProgressProto
from streamlit.string_util import clean_text
def _check_float_between(value: float, low: float=0.0, high: float=1.0) -> bool:
    """
    Checks given value is 'between' the bounds of [low, high],
    considering close values around bounds are acceptable input

    Notes
    -----
    This check is required for handling values that are slightly above or below the
    acceptable range, for example -0.0000000000021, 1.0000000000000013.
    These values are little off the conventional 0.0 <= x <= 1.0 condition
    due to floating point operations, but should still be considered acceptable input.

    Parameters
    ----------
    value : float
    low : float
    high : float

    """
    return low <= value <= high or math.isclose(value, low, rel_tol=1e-09, abs_tol=1e-09) or math.isclose(value, high, rel_tol=1e-09, abs_tol=1e-09)