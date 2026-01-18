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
class MergedPartialArguments(AbstractArguments):

    def __init__(self, partial_arguments, call_arguments, instance=None):
        self._partial_arguments = partial_arguments
        self._call_arguments = call_arguments
        self._instance = instance

    def unpack(self, funcdef=None):
        unpacked = self._partial_arguments.unpack(funcdef)
        next(unpacked, None)
        if self._instance is not None:
            yield (None, LazyKnownValue(self._instance))
        for key_lazy_value in unpacked:
            yield key_lazy_value
        for key_lazy_value in self._call_arguments.unpack(funcdef):
            yield key_lazy_value