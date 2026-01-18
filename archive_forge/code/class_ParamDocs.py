import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies
class ParamDocs(dict):
    """
    Represents a set of parameter documents in the docstring.
    """

    def __repr__(self):
        return f'ParamDocs({super().__repr__()})'

    def format(self, **kwargs):
        """
        Formats values to be substituted in via the format_docstring() method.

        Args:
            kwargs: A `dict` in the form of `{"< placeholder name >": "< value >"}`.

        Returns:
            A new `ParamDocs` instance with the formatted param docs.

        .. code-block:: text
            :caption: Example

            >>> pd = ParamDocs(p1="{{ doc1 }}", p2="{{ doc2 }}")
            >>> pd.format(doc1="foo", doc2="bar")
            ParamDocs({'p1': 'foo', 'p2': 'bar'})
        """
        replacements = _replace_keys_with_placeholders(kwargs)
        return ParamDocs({k: _replace_all(v, replacements) for k, v in self.items()})

    def format_docstring(self, docstring: str) -> str:
        """
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
        """
        if docstring is None:
            return None
        replacements = _replace_keys_with_placeholders(self)
        lines = docstring.splitlines()
        for i, line in enumerate(lines):
            lines[i] = _replace_all(line, replacements)
        return '\n'.join(lines)