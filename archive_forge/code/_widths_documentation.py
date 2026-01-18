import pandas as pd
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict

    Given a DataFrame, with all values and the Index as floats,
    and given a float key, find the row that matches the key, or 
    find the two rows surrounding that key, and return the interpolated
    value for the specified column, based on where the key falls between
    the two rows.  If they key is an exact match for a key in the index,
    the return the exact value from the column.  If the key is less than
    or greater than any key in the index, then return either the first
    or last value for the column.
    