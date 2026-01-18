from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def add_compilation_cache(url: str, data: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Seeds compilation cache for given url. Compilation cache does not survive
    cross-process navigation.

    **EXPERIMENTAL**

    :param url:
    :param data: Base64-encoded data
    """
    params: T_JSON_DICT = dict()
    params['url'] = url
    params['data'] = data
    cmd_dict: T_JSON_DICT = {'method': 'Page.addCompilationCache', 'params': params}
    json = (yield cmd_dict)