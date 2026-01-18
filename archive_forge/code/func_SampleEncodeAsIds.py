from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def SampleEncodeAsIds(self, input, nbest_size=None, alpha=None, **kwargs):
    return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha, out_type=int, enable_sampling=True, **kwargs)