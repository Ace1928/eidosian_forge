from __future__ import annotations
import re
import warnings
from traitlets.log import get_logger
from nbformat import v3 as _v_latest
from nbformat.v3 import (
from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, validate
def _warn_format():
    warnings.warn('Non-JSON file support in nbformat is deprecated since nbformat 1.0.\n    Use nbconvert to create files of other formats.', stacklevel=2)