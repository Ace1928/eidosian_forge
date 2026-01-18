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
class NewTypeFunction(ValueWrapper):

    def py__call__(self, arguments):
        ordered_args = arguments.unpack()
        next(ordered_args, (None, None))
        _, second_arg = next(ordered_args, (None, None))
        if second_arg is None:
            return NO_VALUES
        return ValueSet((NewType(self.inference_state, contextualized_node.context, contextualized_node.node, second_arg.infer()) for contextualized_node in arguments.get_calling_nodes()))