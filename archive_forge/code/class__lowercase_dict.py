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
class _lowercase_dict(_lowercase_dict_base):
    """Dictionary wrapper which lowercase keys upon lookup."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key.lower())