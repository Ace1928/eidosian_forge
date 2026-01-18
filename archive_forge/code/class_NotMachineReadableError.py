import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class NotMachineReadableError(Error):
    """Raised when the input is not a machine-readable debian/copyright file."""