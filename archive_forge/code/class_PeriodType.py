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
class PeriodType(pa.ExtensionType):

    def __init__(self, freq):
        self._freq = freq
        pa.ExtensionType.__init__(self, pa.int64(), 'test.period')

    @property
    def freq(self):
        return self._freq

    def __arrow_ext_serialize__(self):
        return 'freq={}'.format(self.freq).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        serialized = serialized.decode()
        assert serialized.startswith('freq=')
        freq = serialized.split('=')[1]
        return PeriodType(freq)

    def __eq__(self, other):
        if isinstance(other, pa.BaseExtensionType):
            return isinstance(self, type(other)) and self.freq == other.freq
        else:
            return NotImplemented