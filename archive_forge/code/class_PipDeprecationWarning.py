import logging
import warnings
from typing import Any, Optional, TextIO, Type, Union
from pip._vendor.packaging.version import parse
from pip import __version__ as current_version  # NOTE: tests patch this name.
class PipDeprecationWarning(Warning):
    pass