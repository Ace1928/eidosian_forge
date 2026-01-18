import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
@st.composite
def _pymap(draw, key_type, value_type, size, nullable=True):
    length = draw(size)
    keys = draw(_pylist(key_type, size=length, nullable=False))
    values = draw(_pylist(value_type, size=length, nullable=nullable))
    return list(zip(keys, values))