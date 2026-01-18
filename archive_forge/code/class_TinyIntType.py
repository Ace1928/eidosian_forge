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
class TinyIntType(pa.ExtensionType):

    def __init__(self):
        super().__init__(pa.int8(), 'pyarrow.tests.TinyIntType')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        assert serialized == b''
        assert storage_type == pa.int8()
        return cls()