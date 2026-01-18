import inspect
from pathlib import Path
from jedi.parser_utils import get_cached_code_lines
from jedi import settings
from jedi.cache import memoize_method
from jedi.inference import compiled
from jedi.file_io import FileIO
from jedi.inference.names import NameWrapper
from jedi.inference.base_value import ValueSet, ValueWrapper, NO_VALUES
from jedi.inference.value import ModuleValue
from jedi.inference.cache import inference_state_function_cache, \
from jedi.inference.compiled.access import ALLOWED_GETITEM_TYPES, get_api_type
from jedi.inference.gradual.conversion import to_stub
from jedi.inference.context import CompiledContext, CompiledModuleContext, \
class MixedObject(ValueWrapper):
    """
    A ``MixedObject`` is used in two ways:

    1. It uses the default logic of ``parser.python.tree`` objects,
    2. except for getattr calls and signatures. The names dicts are generated
       in a fashion like ``CompiledValue``.

    This combined logic makes it possible to provide more powerful REPL
    completion. It allows side effects that are not noticable with the default
    parser structure to still be completable.

    The biggest difference from CompiledValue to MixedObject is that we are
    generally dealing with Python code and not with C code. This will generate
    fewer special cases, because we in Python you don't have the same freedoms
    to modify the runtime.
    """

    def __init__(self, compiled_value, tree_value):
        super().__init__(tree_value)
        self.compiled_value = compiled_value
        self.access_handle = compiled_value.access_handle

    def get_filters(self, *args, **kwargs):
        yield MixedObjectFilter(self.inference_state, self.compiled_value, self._wrapped_value)

    def get_signatures(self):
        return self.compiled_value.get_signatures()

    @inference_state_method_cache(default=NO_VALUES)
    def py__call__(self, arguments):
        values = to_stub(self._wrapped_value)
        if not values:
            values = self._wrapped_value
        return values.py__call__(arguments)

    def get_safe_value(self, default=_sentinel):
        if default is _sentinel:
            return self.compiled_value.get_safe_value()
        else:
            return self.compiled_value.get_safe_value(default)

    @property
    def array_type(self):
        return self.compiled_value.array_type

    def get_key_values(self):
        return self.compiled_value.get_key_values()

    def py__simple_getitem__(self, index):
        python_object = self.compiled_value.access_handle.access._obj
        if type(python_object) in ALLOWED_GETITEM_TYPES:
            return self.compiled_value.py__simple_getitem__(index)
        return self._wrapped_value.py__simple_getitem__(index)

    def negate(self):
        return self.compiled_value.negate()

    def _as_context(self):
        if self.parent_context is None:
            return MixedModuleContext(self)
        return MixedContext(self)

    def __repr__(self):
        return '<%s: %s; %s>' % (type(self).__name__, self.access_handle.get_repr(), self._wrapped_value)