import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
class _MissingDynamic(SetuptoolsWarning):
    _SUMMARY = '`{field}` defined outside of `pyproject.toml` is ignored.'
    _DETAILS = '\n    The following seems to be defined outside of `pyproject.toml`:\n\n    `{field} = {value!r}`\n\n    According to the spec (see the link below), however, setuptools CANNOT\n    consider this value unless `{field}` is listed as `dynamic`.\n\n    https://packaging.python.org/en/latest/specifications/pyproject-toml/#declaring-project-metadata-the-project-table\n\n    To prevent this problem, you can list `{field}` under `dynamic` or alternatively\n    remove the `[project]` table from your file and rely entirely on other means of\n    configuration.\n    '

    @classmethod
    def details(cls, field: str, value: Any) -> str:
        return cls._DETAILS.format(field=field, value=value)