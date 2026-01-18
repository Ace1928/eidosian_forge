import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from itertools import chain
from types import MappingProxyType
from typing import (
from ..warnings import SetuptoolsWarning, SetuptoolsDeprecationWarning
class _WouldIgnoreField(SetuptoolsDeprecationWarning):
    _SUMMARY = '`{field}` defined outside of `pyproject.toml` would be ignored.'
    _DETAILS = '\n    ##########################################################################\n    # configuration would be ignored/result in error due to `pyproject.toml` #\n    ##########################################################################\n\n    The following seems to be defined outside of `pyproject.toml`:\n\n    `{field} = {value!r}`\n\n    According to the spec (see the link below), however, setuptools CANNOT\n    consider this value unless `{field}` is listed as `dynamic`.\n\n    https://packaging.python.org/en/latest/specifications/declaring-project-metadata/\n\n    For the time being, `setuptools` will still consider the given value (as a\n    **transitional** measure), but please note that future releases of setuptools will\n    follow strictly the standard.\n\n    To prevent this warning, you can list `{field}` under `dynamic` or alternatively\n    remove the `[project]` table from your file and rely entirely on other means of\n    configuration.\n    '
    _DUE_DATE = (2023, 10, 30)