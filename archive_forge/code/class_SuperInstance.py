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
class SuperInstance(LazyValueWrapper):
    """To be used like the object ``super`` returns."""

    def __init__(self, inference_state, instance):
        self.inference_state = inference_state
        self._instance = instance

    def _get_bases(self):
        return self._instance.py__class__().py__bases__()

    def _get_wrapped_value(self):
        objs = self._get_bases()[0].infer().execute_with_values()
        if not objs:
            return self._instance
        return next(iter(objs))

    def get_filters(self, origin_scope=None):
        for b in self._get_bases():
            for value in b.infer().execute_with_values():
                for f in value.get_filters():
                    yield f