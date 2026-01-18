import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
    """Reset dataclass_transform_spec attribute of ModelMetaclass.

        Let the plugin handle it. This behavior can be disabled
        if 'debug_dataclass_transform' is set to True', for testing purposes.
        """
    if self.plugin_config.debug_dataclass_transform:
        return
    info_metaclass = ctx.cls.info.declared_metaclass
    assert info_metaclass, "callback not passed from 'get_metaclass_hook'"
    if getattr(info_metaclass.type, 'dataclass_transform_spec', None):
        info_metaclass.type.dataclass_transform_spec = None