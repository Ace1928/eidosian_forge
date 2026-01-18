from __future__ import annotations
from collections import (
import copy
from typing import (
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame
def _normalise_json(data: Any, key_string: str, normalized_dict: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f'{key_string}{separator}{key}'
            if not key_string:
                new_key = new_key.removeprefix(separator)
            _normalise_json(data=value, key_string=new_key, normalized_dict=normalized_dict, separator=separator)
    else:
        normalized_dict[key_string] = data
    return normalized_dict