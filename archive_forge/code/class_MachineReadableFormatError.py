import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class MachineReadableFormatError(Error, ValueError):
    """Raised when the input is not valid.

    This is both a `copyright.Error` and a `ValueError` to ease handling of
    errors coming from this module.
    """