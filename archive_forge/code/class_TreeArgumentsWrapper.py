import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
class TreeArgumentsWrapper(_AbstractArgumentsMixin):

    def __init__(self, arguments):
        self._wrapped_arguments = arguments

    @property
    def context(self):
        return self._wrapped_arguments.context

    @property
    def argument_node(self):
        return self._wrapped_arguments.argument_node

    @property
    def trailer(self):
        return self._wrapped_arguments.trailer

    def unpack(self, func=None):
        raise NotImplementedError

    def get_calling_nodes(self):
        return self._wrapped_arguments.get_calling_nodes()

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._wrapped_arguments)