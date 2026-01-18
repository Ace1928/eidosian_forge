from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def EncodeAsSerializedProto(self, input, **kwargs):
    return self.Encode(input=input, out_type='serialized_proto', **kwargs)