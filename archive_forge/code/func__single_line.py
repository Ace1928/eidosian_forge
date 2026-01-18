import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def _single_line(s):
    """Returns s if it is a single line; otherwise raises MachineReadableFormatError."""
    if '\n' in s:
        raise MachineReadableFormatError('must be single line')
    return s