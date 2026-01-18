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
def _get_modin_deps_info() -> Mapping[str, Optional[JSONSerializable]]:
    """
    Return Modin-specific dependencies information as a JSON serializable dictionary.

    Returns
    -------
    Mapping[str, Optional[pandas.JSONSerializable]]
        The dictionary of Modin dependencies and their versions.
    """
    import modin
    result = {'modin': modin.__version__}
    for pkg_name, pkg_version in [('ray', MIN_RAY_VERSION), ('dask', MIN_DASK_VERSION), ('distributed', MIN_DASK_VERSION)]:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            result[pkg_name] = None
        else:
            result[pkg_name] = pkg.__version__ + (f' (outdated; >={pkg_version} required)' if version.parse(pkg.__version__) < pkg_version else '')
    try:
        from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import DbWorker
        result['hdk'] = 'present'
    except ImportError:
        result['hdk'] = None
    return result