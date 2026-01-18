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
class AnnotatedType(pa.ExtensionType):
    """
    Generic extension type that can store any storage type.
    """

    def __init__(self, storage_type, annotation):
        self.annotation = annotation
        super().__init__(storage_type, 'pyarrow.tests.AnnotatedType')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        assert serialized == b''
        return cls(storage_type)