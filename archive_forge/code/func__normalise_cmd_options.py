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
def _normalise_cmd_options(desc: List[Tuple[str, Optional[str], str]]) -> Set[str]:
    return {_normalise_cmd_option_key(fancy_option[0]) for fancy_option in desc}