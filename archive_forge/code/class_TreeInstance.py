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
class TreeInstance(_BaseTreeInstance):

    def __init__(self, inference_state, parent_context, class_value, arguments):
        if class_value.py__name__() in ['list', 'set'] and parent_context.get_root_context().is_builtins_module():
            if settings.dynamic_array_additions:
                arguments = get_dynamic_array_instance(self, arguments)
        super().__init__(inference_state, parent_context, class_value)
        self._arguments = arguments
        self.tree_node = class_value.tree_node

    @inference_state_method_cache(default=None)
    def _get_annotated_class_object(self):
        from jedi.inference.gradual.annotation import py__annotations__, infer_type_vars_for_execution
        args = InstanceArguments(self, self._arguments)
        for signature in self.class_value.py__getattribute__('__init__').get_signatures():
            funcdef = signature.value.tree_node
            if funcdef is None or funcdef.type != 'funcdef' or (not signature.matches_signature(args)):
                continue
            bound_method = BoundMethod(self, self.class_value.as_context(), signature.value)
            all_annotations = py__annotations__(funcdef)
            type_var_dict = infer_type_vars_for_execution(bound_method, args, all_annotations)
            if type_var_dict:
                defined, = self.class_value.define_generics(infer_type_vars_for_execution(signature.value, args, all_annotations))
                debug.dbg('Inferred instance value as %s', defined, color='BLUE')
                return defined
        return None

    def get_annotated_class_object(self):
        return self._get_annotated_class_object() or self.class_value

    def get_key_values(self):
        values = NO_VALUES
        if self.array_type == 'dict':
            for i, (key, instance) in enumerate(self._arguments.unpack()):
                if key is None and i == 0:
                    values |= ValueSet.from_sets((v.get_key_values() for v in instance.infer() if v.array_type == 'dict'))
                if key:
                    values |= ValueSet([compiled.create_simple_object(self.inference_state, key)])
        return values

    def py__simple_getitem__(self, index):
        if self.array_type == 'dict':
            for key, lazy_context in reversed(list(self._arguments.unpack())):
                if key is None:
                    values = ValueSet.from_sets((dct_value.py__simple_getitem__(index) for dct_value in lazy_context.infer() if dct_value.array_type == 'dict'))
                    if values:
                        return values
                elif key == index:
                    return lazy_context.infer()
        return super().py__simple_getitem__(index)

    def __repr__(self):
        return '<%s of %s(%s)>' % (self.__class__.__name__, self.class_value, self._arguments)