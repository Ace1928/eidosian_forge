import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def _get_array_value(self, field):
    if field not in self:
        raise KeyError("'{}' not found in buildinfo".format(field))
    return list(self[field].replace('\n', '').strip().split())