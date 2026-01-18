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
class PeriodTypeWithClass(PeriodType):

    def __init__(self, freq):
        PeriodType.__init__(self, freq)

    def __arrow_ext_class__(self):
        return PeriodArray

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        freq = PeriodType.__arrow_ext_deserialize__(storage_type, serialized).freq
        return PeriodTypeWithClass(freq)