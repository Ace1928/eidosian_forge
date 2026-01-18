import itertools
from jedi import debug
from jedi.inference.compiled import builtin_from_name, create_simple_object
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.arguments import repack_with_argument_clinic
from jedi.inference.filters import FilterWrapper
from jedi.inference.names import NameWrapper, ValueName
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import BaseTypingValue, \
from jedi.inference.gradual.type_var import TypeVarClass
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
class TypingModuleName(NameWrapper):

    def infer(self):
        return ValueSet(self._remap())

    def _remap(self):
        name = self.string_name
        inference_state = self.parent_context.inference_state
        try:
            actual = _TYPE_ALIAS_TYPES[name]
        except KeyError:
            pass
        else:
            yield TypeAlias.create_cached(inference_state, self.parent_context, self.tree_name, actual)
            return
        if name in _PROXY_CLASS_TYPES:
            yield ProxyTypingClassValue.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name in _PROXY_TYPES:
            yield ProxyTypingValue.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'runtime':
            return
        elif name == 'TypeVar':
            cls, = self._wrapped_name.infer()
            yield TypeVarClass.create_cached(inference_state, cls)
        elif name == 'Any':
            yield AnyClass.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'TYPE_CHECKING':
            yield builtin_from_name(inference_state, 'True')
        elif name == 'overload':
            yield OverloadFunction.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'NewType':
            v, = self._wrapped_name.infer()
            yield NewTypeFunction.create_cached(inference_state, v)
        elif name == 'cast':
            cast_fn, = self._wrapped_name.infer()
            yield CastFunction.create_cached(inference_state, cast_fn)
        elif name == 'TypedDict':
            yield TypedDictClass.create_cached(inference_state, self.parent_context, self.tree_name)
        else:
            yield from self._wrapped_name.infer()