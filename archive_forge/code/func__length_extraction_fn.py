import logging
import uuid
from abc import ABC
from copy import copy
import pandas
from pandas.api.types import is_scalar
from pandas.util import cache_readonly
from modin.core.storage_formats.pandas.utils import length_fn_pandas, width_fn_pandas
from modin.logging import ClassLogger, get_logger
from modin.pandas.indexing import compute_sliced_len
@classmethod
def _length_extraction_fn(cls):
    """
        Return the function that computes the length of the object wrapped by this partition.

        Returns
        -------
        callable
            The function that computes the length of the object wrapped by this partition.
        """
    return length_fn_pandas