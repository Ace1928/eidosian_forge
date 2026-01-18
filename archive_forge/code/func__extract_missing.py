from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _extract_missing(values):
    """Extract missing values from `values`.

    Parameters
    ----------
    values: set
        Set of values to extract missing from.

    Returns
    -------
    output: set
        Set with missing values extracted.

    missing_values: MissingValues
        Object with missing value information.
    """
    missing_values_set = {value for value in values if value is None or is_scalar_nan(value)}
    if not missing_values_set:
        return (values, MissingValues(nan=False, none=False))
    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)
    output = values - missing_values_set
    return (output, output_missing_values)