import json
from array import array
from enum import Enum, auto
from typing import Any
def _copytobuffer_return_scalar(xxx: Any) -> tuple[array, DataType]:
    """
    Prepares scalar for PROJ C-API:
    - Makes a copy because PROJ modifies buffer in place
    - Make sure dtype is double as that is what PROJ expects
    - Makes sure object supports Python Buffer API

    Parameters
    -----------
    xxx: float or 0-d numpy array

    Returns
    -------
    tuple[Any, DataType]
        The copy of the data prepared for the PROJ API & Python Buffer API.
    """
    try:
        return (array('d', (float(xxx),)), DataType.FLOAT)
    except Exception:
        raise TypeError('input must be a scalar') from None