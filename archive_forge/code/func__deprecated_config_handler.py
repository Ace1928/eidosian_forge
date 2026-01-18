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
def _deprecated_config_handler(self, func, msg, **kw):
    """this function will wrap around parameters that are deprecated

        :param msg: deprecation message
        :param func: function to be wrapped around
        """

    @wraps(func)
    def config_handler(*args, **kwargs):
        kw.setdefault('stacklevel', 2)
        _DeprecatedConfig.emit('Deprecated config in `setup.cfg`', msg, **kw)
        return func(*args, **kwargs)
    return config_handler