import re
from functools import partial
from inspect import Parameter
from pathlib import Path
from typing import Optional
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, \
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
class CompiledModule(CompiledValue):
    file_io = None

    def _as_context(self):
        return CompiledModuleContext(self)

    def py__path__(self):
        return self.access_handle.py__path__()

    def is_package(self):
        return self.py__path__() is not None

    @property
    def string_names(self):
        name = self.py__name__()
        if name is None:
            return ()
        return tuple(name.split('.'))

    def py__file__(self) -> Optional[Path]:
        return self.access_handle.py__file__()