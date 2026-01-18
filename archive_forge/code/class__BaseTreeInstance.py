from abc import abstractproperty
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.inference import compiled
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.helpers import values_from_qualified_names, is_big_annoying_library
from jedi.inference.filters import AbstractFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, TreeNameDefinition, ParamName, \
from jedi.inference.base_value import Value, NO_VALUES, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.value.function import \
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod
class _BaseTreeInstance(AbstractInstanceValue):

    @property
    def array_type(self):
        name = self.class_value.py__name__()
        if name in ['list', 'set', 'dict'] and self.parent_context.get_root_context().is_builtins_module():
            return name
        return None

    @property
    def name(self):
        return ValueName(self, self.class_value.name.tree_name)

    def get_filters(self, origin_scope=None, include_self_names=True):
        class_value = self.get_annotated_class_object()
        if include_self_names:
            for cls in class_value.py__mro__():
                if not cls.is_compiled():
                    yield SelfAttributeFilter(self, class_value, cls.as_context(), origin_scope)
        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)
        for f in class_filters:
            if isinstance(f, ClassFilter):
                yield InstanceClassFilter(self, f)
            elif isinstance(f, CompiledValueFilter):
                yield CompiledInstanceClassFilter(self, f)
            else:
                yield f

    @inference_state_method_cache()
    def create_instance_context(self, class_context, node):
        new = node
        while True:
            func_node = new
            new = search_ancestor(new, 'funcdef', 'classdef')
            if class_context.tree_node is new:
                func = FunctionValue.from_context(class_context, func_node)
                bound_method = BoundMethod(self, class_context, func)
                if func_node.name.value == '__init__':
                    context = bound_method.as_context(self._arguments)
                else:
                    context = bound_method.as_context()
                break
        return context.create_context(node)

    def py__getattribute__alternatives(self, string_name):
        """
        Since nothing was inferred, now check the __getattr__ and
        __getattribute__ methods. Stubs don't need to be checked, because
        they don't contain any logic.
        """
        if self.is_stub():
            return NO_VALUES
        name = compiled.create_simple_object(self.inference_state, string_name)
        if is_big_annoying_library(self.parent_context):
            return NO_VALUES
        names = self.get_function_slot_names('__getattr__') or self.get_function_slot_names('__getattribute__')
        return self.execute_function_slots(names, name)

    def py__next__(self, contextualized_node=None):
        name = u'__next__'
        next_slot_names = self.get_function_slot_names(name)
        if next_slot_names:
            yield LazyKnownValues(self.execute_function_slots(next_slot_names))
        else:
            debug.warning('Instance has no __next__ function in %s.', self)

    def py__call__(self, arguments):
        names = self.get_function_slot_names('__call__')
        if not names:
            return super().py__call__(arguments)
        return ValueSet.from_sets((name.infer().execute(arguments) for name in names))

    def py__get__(self, instance, class_value):
        """
        obj may be None.
        """
        for cls in self.class_value.py__mro__():
            result = cls.py__get__on_class(self, instance, class_value)
            if result is not NotImplemented:
                return result
        names = self.get_function_slot_names('__get__')
        if names:
            if instance is None:
                instance = compiled.builtin_from_name(self.inference_state, 'None')
            return self.execute_function_slots(names, instance, class_value)
        else:
            return ValueSet([self])