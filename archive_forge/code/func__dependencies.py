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
def _dependencies(dist: 'Distribution', val: list, _root_dir):
    if getattr(dist, 'install_requires', []):
        msg = '`install_requires` overwritten in `pyproject.toml` (dependencies)'
        SetuptoolsWarning.emit(msg)
    dist.install_requires = val