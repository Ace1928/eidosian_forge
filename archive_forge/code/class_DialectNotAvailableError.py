from __future__ import absolute_import, division, print_function
import csv
from io import BytesIO, StringIO
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
class DialectNotAvailableError(Exception):
    pass