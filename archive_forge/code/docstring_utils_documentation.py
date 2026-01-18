import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies

        Formats placeholders in `docstring`.

        Args:
            p1: {{ p1 }}
            p2: {{ p2 }}

        .. code-block:: text
            :caption: Example

            >>> pd = ParamDocs(p1="doc1", p2="doc2
            doc2 second line")
            >>> docstring = '''
            ... Args:
            ...     p1: {{ p1 }}
            ...     p2: {{ p2 }}
            ... '''.strip()
            >>> print(pd.format_docstring(docstring))
        