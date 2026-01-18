from collections import defaultdict
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, \
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
class ExecutedParamName(ParamName):

    def __init__(self, function_value, arguments, param_node, lazy_value, is_default=False):
        super().__init__(function_value, param_node.name, arguments=arguments)
        self._lazy_value = lazy_value
        self._is_default = is_default

    def infer(self):
        return self._lazy_value.infer()

    def matches_signature(self):
        if self._is_default:
            return True
        argument_values = self.infer().py__class__()
        if self.get_kind() in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            return True
        annotations = self.infer_annotation(execute_annotation=False)
        if not annotations:
            return True
        matches = any((c1.is_sub_class_of(c2) for c1 in argument_values for c2 in annotations.gather_annotation_classes()))
        debug.dbg('param compare %s: %s <=> %s', matches, argument_values, annotations, color='BLUE')
        return matches

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.string_name)