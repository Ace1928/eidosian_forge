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
def _get_option(target_obj: Target, key: str):
    """
    Given a target object and option key, get that option from
    the target object, either through a get_{key} method or
    from an attribute directly.
    """
    getter_name = f'get_{key}'
    by_attribute = functools.partial(getattr, target_obj, key)
    getter = getattr(target_obj, getter_name, by_attribute)
    return getter()