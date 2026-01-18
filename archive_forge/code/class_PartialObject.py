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
class PartialObject(ValueWrapper):

    def __init__(self, actual_value, arguments, instance=None):
        super().__init__(actual_value)
        self._arguments = arguments
        self._instance = instance

    def _get_functions(self, unpacked_arguments):
        key, lazy_value = next(unpacked_arguments, (None, None))
        if key is not None or lazy_value is None:
            debug.warning('Partial should have a proper function %s', self._arguments)
            return None
        return lazy_value.infer()

    def get_signatures(self):
        unpacked_arguments = self._arguments.unpack()
        funcs = self._get_functions(unpacked_arguments)
        if funcs is None:
            return []
        arg_count = 0
        if self._instance is not None:
            arg_count = 1
        keys = set()
        for key, _ in unpacked_arguments:
            if key is None:
                arg_count += 1
            else:
                keys.add(key)
        return [PartialSignature(s, arg_count, keys) for s in funcs.get_signatures()]

    def py__call__(self, arguments):
        funcs = self._get_functions(self._arguments.unpack())
        if funcs is None:
            return NO_VALUES
        return funcs.execute(MergedPartialArguments(self._arguments, arguments, self._instance))

    def py__doc__(self):
        """
        In CPython partial does not replace the docstring. However we are still
        imitating it here, because we want this docstring to be worth something
        for the user.
        """
        callables = self._get_functions(self._arguments.unpack())
        if callables is None:
            return ''
        for callable_ in callables:
            return callable_.py__doc__()
        return ''

    def py__get__(self, instance, class_value):
        return ValueSet([self])