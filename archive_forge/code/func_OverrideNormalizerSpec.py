from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def OverrideNormalizerSpec(self, **kwargs):
    new_kwargs = {}
    for key, value in kwargs.items():
        new_kwargs[key] = str(value)
    return self._OverrideNormalizerSpec(new_kwargs)