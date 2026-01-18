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
class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type = 'property'

    def __init__(self, property_obj, function):
        super().__init__(property_obj)
        self._function = function

    def py__get__(self, instance, class_value):
        if instance is None:
            return ValueSet([self])
        return self._function.execute_with_values(instance)

    @publish_method('deleter')
    @publish_method('getter')
    @publish_method('setter')
    def _return_self(self, arguments):
        return ValueSet({self})