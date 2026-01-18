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
def _project_urls(dist: 'Distribution', val: dict, _root_dir):
    _set_config(dist, 'project_urls', val)