import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
class PeriodTypeWithToPandasDtype(PeriodType):

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        freq = PeriodType.__arrow_ext_deserialize__(storage_type, serialized).freq
        return PeriodTypeWithToPandasDtype(freq)

    def to_pandas_dtype(self):
        import pandas as pd
        return pd.PeriodDtype(freq=self.freq)