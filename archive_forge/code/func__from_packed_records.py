import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
@classmethod
def _from_packed_records(cls, records, **kwargs):
    """Helper function for `_from_json_dict_` to construct from packed records."""
    return cls(records={key: _unpack_digits(**val) for key, val in records.items()}, **kwargs)