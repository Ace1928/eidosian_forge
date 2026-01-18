import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def _complete_python(self, leaf):
    """
        Analyzes the current context of a completion and decides what to
        return.

        Technically this works by generating a parser stack and analysing the
        current stack for possible grammar nodes.

        Possible enhancements:
        - global/nonlocal search global
        - yield from / raise from <- could be only exceptions/generators
        - In args: */**: no completion
        - In params (also lambda): no completion before =
        """
    grammar = self._inference_state.grammar
    self.stack = stack = None
    self._position = (self._original_position[0], self._original_position[1] - len(self._like_name))
    cached_name = None
    try:
        self.stack = stack = helpers.get_stack_at_position(grammar, self._code_lines, leaf, self._position)
    except helpers.OnErrorLeaf as e:
        value = e.error_leaf.value
        if value == '.':
            return (cached_name, [])
        return (cached_name, self._complete_global_scope())
    allowed_transitions = list(stack._allowed_transition_names_and_token_types())
    if 'if' in allowed_transitions:
        leaf = self._module_node.get_leaf_for_position(self._position, include_prefixes=True)
        previous_leaf = leaf.get_previous_leaf()
        indent = self._position[1]
        if not leaf.start_pos <= self._position <= leaf.end_pos:
            indent = leaf.start_pos[1]
        if previous_leaf is not None:
            stmt = previous_leaf
            while True:
                stmt = search_ancestor(stmt, 'if_stmt', 'for_stmt', 'while_stmt', 'try_stmt', 'error_node')
                if stmt is None:
                    break
                type_ = stmt.type
                if type_ == 'error_node':
                    first = stmt.children[0]
                    if isinstance(first, Leaf):
                        type_ = first.value + '_stmt'
                if stmt.start_pos[1] == indent:
                    if type_ == 'if_stmt':
                        allowed_transitions += ['elif', 'else']
                    elif type_ == 'try_stmt':
                        allowed_transitions += ['except', 'finally', 'else']
                    elif type_ == 'for_stmt':
                        allowed_transitions.append('else')
    completion_names = []
    kwargs_only = False
    if any((t in allowed_transitions for t in (PythonTokenTypes.NAME, PythonTokenTypes.INDENT))):
        nonterminals = [stack_node.nonterminal for stack_node in stack]
        nodes = _gather_nodes(stack)
        if nodes and nodes[-1] in ('as', 'def', 'class'):
            return (cached_name, list(self._complete_inherited(is_function=True)))
        elif 'import_stmt' in nonterminals:
            level, names = parse_dotted_names(nodes, 'import_from' in nonterminals)
            only_modules = not ('import_from' in nonterminals and 'import' in nodes)
            completion_names += self._get_importer_names(names, level, only_modules=only_modules)
        elif nonterminals[-1] in ('trailer', 'dotted_name') and nodes[-1] == '.':
            dot = self._module_node.get_leaf_for_position(self._position)
            if dot.type == 'endmarker':
                dot = leaf.get_previous_leaf()
            cached_name, n = self._complete_trailer(dot.get_previous_leaf())
            completion_names += n
        elif self._is_parameter_completion():
            completion_names += self._complete_params(leaf)
        else:
            if nodes[-1] in ['(', ','] and nonterminals[-1] in ('trailer', 'arglist', 'decorator'):
                signatures = self._signatures_callback(*self._position)
                if signatures:
                    call_details = signatures[0]._call_details
                    used_kwargs = list(call_details.iter_used_keyword_arguments())
                    positional_count = call_details.count_positional_arguments()
                    completion_names += _get_signature_param_names(signatures, positional_count, used_kwargs)
                    kwargs_only = _must_be_kwarg(signatures, positional_count, used_kwargs)
            if not kwargs_only:
                completion_names += self._complete_global_scope()
                completion_names += self._complete_inherited(is_function=False)
    if not kwargs_only:
        current_line = self._code_lines[self._position[0] - 1][:self._position[1]]
        completion_names += self._complete_keywords(allowed_transitions, only_values=not (not current_line or (current_line[-1] in ' \t.;' and current_line[-3:] != '...')))
    return (cached_name, completion_names)