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
def _license(dist: 'Distribution', val: dict, root_dir: _Path):
    from setuptools.config import expand
    if 'file' in val:
        _set_config(dist, 'license', expand.read_files([val['file']], root_dir))
        dist._referenced_files.add(val['file'])
    else:
        _set_config(dist, 'license', val['text'])