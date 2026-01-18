import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@lru_cache
def _compile_jinja_template(self, chat_template):
    try:
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError('apply_chat_template requires jinja2 to be installed.')
    if version.parse(jinja2.__version__) <= version.parse('3.0.0'):
        raise ImportError(f'apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is {jinja2.__version__}.')

    def raise_exception(message):
        raise TemplateError(message)
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals['raise_exception'] = raise_exception
    return jinja_env.from_string(chat_template)