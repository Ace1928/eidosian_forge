import parso
import os
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import safe_property
from jedi.inference.helpers import get_str_or_none
from jedi.inference.arguments import iterate_argument_clinic, ParamIssue, \
from jedi.inference import analysis
from jedi.inference import compiled
from jedi.inference.value.instance import \
from jedi.inference.base_value import ContextualizedNode, \
from jedi.inference.value import ClassValue, ModuleValue
from jedi.inference.value.klass import ClassMixin
from jedi.inference.value.function import FunctionMixin
from jedi.inference.value import iterable
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValue, \
from jedi.inference.names import ValueName, BaseTreeParamName
from jedi.inference.filters import AttributeOverwrite, publish_method, \
from jedi.inference.signature import AbstractSignature, SignatureWrapper
from operator import itemgetter as _itemgetter
from collections import OrderedDict
class EnumInstance(LazyValueWrapper):

    def __init__(self, cls, name):
        self.inference_state = cls.inference_state
        self._cls = cls
        self._name = name
        self.tree_node = self._name.tree_name

    @safe_property
    def name(self):
        return ValueName(self, self._name.tree_name)

    def _get_wrapped_value(self):
        n = self._name.string_name
        if n.startswith('__') and n.endswith('__') or self._name.api_type == 'function':
            inferred = self._name.infer()
            if inferred:
                return next(iter(inferred))
            o, = self.inference_state.builtins_module.py__getattribute__('object')
            return o
        value, = self._cls.execute_with_values()
        return value

    def get_filters(self, origin_scope=None):
        yield DictFilter(dict(name=compiled.create_simple_object(self.inference_state, self._name.string_name).name, value=self._name))
        for f in self._get_wrapped_value().get_filters():
            yield f