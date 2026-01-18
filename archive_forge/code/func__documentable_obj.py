import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def _documentable_obj(obj: object) -> bool:
    """Check if `obj` docstring could be patched."""
    return bool(callable(obj) or (isinstance(obj, property) and obj.fget) or (isinstance(obj, (staticmethod, classmethod)) and obj.__func__))