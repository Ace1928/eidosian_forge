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
def _get_previous_entrypoints(dist: 'Distribution') -> Dict[str, list]:
    ignore = ('console_scripts', 'gui_scripts')
    value = getattr(dist, 'entry_points', None) or {}
    return {k: v for k, v in value.items() if k not in ignore}