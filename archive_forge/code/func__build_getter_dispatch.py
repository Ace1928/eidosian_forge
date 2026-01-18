import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
def _build_getter_dispatch(cfg_obj, cfg_section, converters=None):
    converters = converters or {}
    default_getter = _build_getter(cfg_obj, cfg_section, 'get')
    getters = {int: _build_getter(cfg_obj, cfg_section, 'getint'), bool: _build_getter(cfg_obj, cfg_section, 'getboolean'), float: _build_getter(cfg_obj, cfg_section, 'getfloat'), str: default_getter}
    getters.update({type_: _build_getter(cfg_obj, cfg_section, 'get', converter=converter_fn) for type_, converter_fn in converters.items()})
    return _GetterDispatch(getters, default_getter)