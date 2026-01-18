from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
def _get_from_config_data(key: str) -> bool:
    assert isinstance(config_data, build.ConfigurationData), 'for mypy'
    if key not in config_cache:
        if key in config_data:
            config_cache[key] = bool(config_data.get(key)[0])
        elif strict:
            raise InvalidArguments(f'sourceset.apply: key "{key}" not in passed configuration, and strict set.')
        else:
            config_cache[key] = False
    return config_cache[key]