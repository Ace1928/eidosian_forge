from __future__ import annotations
import functools
import importlib
import pkgutil
import threading
from typing import Any, Callable, Optional, Sequence
import tiktoken_ext
from tiktoken.core import Encoding
def _find_constructors() -> None:
    global ENCODING_CONSTRUCTORS
    with _lock:
        if ENCODING_CONSTRUCTORS is not None:
            return
        ENCODING_CONSTRUCTORS = {}
        for mod_name in _available_plugin_modules():
            mod = importlib.import_module(mod_name)
            try:
                constructors = mod.ENCODING_CONSTRUCTORS
            except AttributeError as e:
                raise ValueError(f'tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS') from e
            for enc_name, constructor in constructors.items():
                if enc_name in ENCODING_CONSTRUCTORS:
                    raise ValueError(f'Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}')
                ENCODING_CONSTRUCTORS[enc_name] = constructor