import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class DEPRECATED(metaclass=PatchClassRepr):
    """Signal value to help with deprecating parameters that use None.

    This is a proxy object, used to signal that a parameter has not been set.
    This is useful if ``None`` is already used for a different purpose or just
    to highlight a deprecated parameter in the signature.
    """