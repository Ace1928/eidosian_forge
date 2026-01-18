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
def _merge_fields(self, s1, s2):
    if not s2:
        return s1
    if not s1:
        return s2
    if self.is_single_line(s1) and self.is_single_line(s2):
        delim = ' '
        if (s1 + s2).count(', '):
            delim = ', '
        L = sorted((s1 + delim + s2).split(delim))
        prev = merged = L[0]
        for item in L[1:]:
            if item == prev:
                continue
            merged = merged + delim + item
            prev = item
        return merged
    if self.is_multi_line(s1) and self.is_multi_line(s2):
        for item in s2.splitlines(True):
            if item not in s1.splitlines(True):
                s1 = s1 + '\n' + item
        return s1
    raise ValueError