import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
@classmethod
def _parse_bool(cls, value):
    """Represents value as boolean.

        :param value:
        :rtype: bool
        """
    value = value.lower()
    return value in ('1', 'true', 'yes')