import copy
import itertools
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
def _infer_subscript_list(context, index):
    """
    Handles slices in subscript nodes.
    """
    if index == ':':
        return ValueSet([iterable.Slice(context, None, None, None)])
    elif index.type == 'subscript' and (not index.children[0] == '.'):
        result = []
        for el in index.children:
            if el == ':':
                if not result:
                    result.append(None)
            elif el.type == 'sliceop':
                if len(el.children) == 2:
                    result.append(el.children[1])
            else:
                result.append(el)
        result += [None] * (3 - len(result))
        return ValueSet([iterable.Slice(context, *result)])
    elif index.type == 'subscriptlist':
        return ValueSet([iterable.SequenceLiteralValue(context.inference_state, context, index)])
    return context.infer_node(index)