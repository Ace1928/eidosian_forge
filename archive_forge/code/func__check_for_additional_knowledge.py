from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
def _check_for_additional_knowledge(self, name_or_str, name_context, position):
    name_context = name_context or self
    if isinstance(name_or_str, Name) and (not name_context.is_instance()):
        flow_scope = name_or_str
        base_nodes = [name_context.tree_node]
        if any((b.type in ('comp_for', 'sync_comp_for') for b in base_nodes)):
            return NO_VALUES
        from jedi.inference.finder import check_flow_information
        while True:
            flow_scope = get_parent_scope(flow_scope, include_flows=True)
            n = check_flow_information(name_context, flow_scope, name_or_str, position)
            if n is not None:
                return n
            if flow_scope in base_nodes:
                break
    return NO_VALUES