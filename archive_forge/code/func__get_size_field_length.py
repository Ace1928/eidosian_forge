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
def _get_size_field_length(self, key):
    if self.size_field_behavior == 'apt-ftparchive':
        return 16
    if self.size_field_behavior == 'dak':
        lengths = [len(str(item['size'])) for item in self[key]]
        return max(lengths)
    raise ValueError('Illegal value for size_field_behavior')