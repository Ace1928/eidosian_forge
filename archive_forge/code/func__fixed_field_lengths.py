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
@property
def _fixed_field_lengths(self):
    fixed_field_lengths = {}
    for key in self._multivalued_fields:
        length = self._get_size_field_length(key)
        fixed_field_lengths[key] = {'size': length}
    return fixed_field_lengths