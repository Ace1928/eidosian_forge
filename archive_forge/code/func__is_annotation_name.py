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
def _is_annotation_name(name):
    ancestor = tree.search_ancestor(name, 'param', 'funcdef', 'expr_stmt')
    if ancestor is None:
        return False
    if ancestor.type in ('param', 'funcdef'):
        ann = ancestor.annotation
        if ann is not None:
            return ann.start_pos <= name.start_pos < ann.end_pos
    elif ancestor.type == 'expr_stmt':
        c = ancestor.children
        if len(c) > 1 and c[1].type == 'annassign':
            return c[1].start_pos <= name.start_pos < c[1].end_pos
    return False