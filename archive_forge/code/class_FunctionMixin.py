from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
class FunctionMixin:
    api_type = 'function'

    def get_filters(self, origin_scope=None):
        cls = self.py__class__()
        for instance in cls.execute_with_values():
            yield from instance.get_filters(origin_scope=origin_scope)

    def py__get__(self, instance, class_value):
        from jedi.inference.value.instance import BoundMethod
        if instance is None:
            return ValueSet([self])
        return ValueSet([BoundMethod(instance, class_value.as_context(), self)])

    def get_param_names(self):
        return [AnonymousParamName(self, param.name) for param in self.tree_node.get_params()]

    @property
    def name(self):
        if self.tree_node.type == 'lambdef':
            return LambdaName(self)
        return ValueName(self, self.tree_node.name)

    def is_function(self):
        return True

    def py__name__(self):
        return self.name.string_name

    def get_type_hint(self, add_class_info=True):
        return_annotation = self.tree_node.annotation
        if return_annotation is None:

            def param_name_to_str(n):
                s = n.string_name
                annotation = n.infer().get_type_hint()
                if annotation is not None:
                    s += ': ' + annotation
                if n.default_node is not None:
                    s += '=' + n.default_node.get_code(include_prefix=False)
                return s
            function_execution = self.as_context()
            result = function_execution.infer()
            return_hint = result.get_type_hint()
            body = self.py__name__() + '(%s)' % ', '.join([param_name_to_str(n) for n in function_execution.get_param_names()])
            if return_hint is None:
                return body
        else:
            return_hint = return_annotation.get_code(include_prefix=False)
            body = self.py__name__() + self.tree_node.children[2].get_code(include_prefix=False)
        return body + ' -> ' + return_hint

    def py__call__(self, arguments):
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def _as_context(self, arguments=None):
        if arguments is None:
            return AnonymousFunctionExecution(self)
        return FunctionExecutionContext(self, arguments)

    def get_signatures(self):
        return [TreeSignature(f) for f in self.get_signature_functions()]