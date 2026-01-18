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
class MyListType(pa.ExtensionType):

    def __init__(self, storage_type):
        assert isinstance(storage_type, pa.ListType)
        super().__init__(storage_type, 'pyarrow.tests.MyListType')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        assert serialized == b''
        return cls(storage_type)