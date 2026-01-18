import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
@typing.no_type_check
def _unicode_safe_getattr(obj, name, default=None):
    warnings.warn('_unicode_safe_getattr is deprecated and will be removed in the next release, use getattr() instead', DeprecationWarning, stacklevel=2)
    return getattr(obj, name, default)