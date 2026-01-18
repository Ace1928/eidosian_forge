from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, cast
from ..error import GraphQLError
from ..language import (
from ..type import (
from ..utilities import TypeInfo, TypeInfoVisitor
class ASTValidationContext:
    """Utility class providing a context for validation of an AST.

    An instance of this class is passed as the context attribute to all Validators,
    allowing access to commonly useful contextual information from within a validation
    rule.
    """
    document: DocumentNode
    _fragments: Optional[Dict[str, FragmentDefinitionNode]]
    _fragment_spreads: Dict[SelectionSetNode, List[FragmentSpreadNode]]
    _recursively_referenced_fragments: Dict[OperationDefinitionNode, List[FragmentDefinitionNode]]

    def __init__(self, ast: DocumentNode, on_error: Callable[[GraphQLError], None]) -> None:
        self.document = ast
        self.on_error = on_error
        self._fragments = None
        self._fragment_spreads = {}
        self._recursively_referenced_fragments = {}

    def on_error(self, error: GraphQLError) -> None:
        pass

    def report_error(self, error: GraphQLError) -> None:
        self.on_error(error)

    def get_fragment(self, name: str) -> Optional[FragmentDefinitionNode]:
        fragments = self._fragments
        if fragments is None:
            fragments = {statement.name.value: statement for statement in self.document.definitions if isinstance(statement, FragmentDefinitionNode)}
            self._fragments = fragments
        return fragments.get(name)

    def get_fragment_spreads(self, node: SelectionSetNode) -> List[FragmentSpreadNode]:
        spreads = self._fragment_spreads.get(node)
        if spreads is None:
            spreads = []
            append_spread = spreads.append
            sets_to_visit = [node]
            append_set = sets_to_visit.append
            pop_set = sets_to_visit.pop
            while sets_to_visit:
                visited_set = pop_set()
                for selection in visited_set.selections:
                    if isinstance(selection, FragmentSpreadNode):
                        append_spread(selection)
                    else:
                        set_to_visit = cast(NodeWithSelectionSet, selection).selection_set
                        if set_to_visit:
                            append_set(set_to_visit)
            self._fragment_spreads[node] = spreads
        return spreads

    def get_recursively_referenced_fragments(self, operation: OperationDefinitionNode) -> List[FragmentDefinitionNode]:
        fragments = self._recursively_referenced_fragments.get(operation)
        if fragments is None:
            fragments = []
            append_fragment = fragments.append
            collected_names: Set[str] = set()
            add_name = collected_names.add
            nodes_to_visit = [operation.selection_set]
            append_node = nodes_to_visit.append
            pop_node = nodes_to_visit.pop
            get_fragment = self.get_fragment
            get_fragment_spreads = self.get_fragment_spreads
            while nodes_to_visit:
                visited_node = pop_node()
                for spread in get_fragment_spreads(visited_node):
                    frag_name = spread.name.value
                    if frag_name not in collected_names:
                        add_name(frag_name)
                        fragment = get_fragment(frag_name)
                        if fragment:
                            append_fragment(fragment)
                            append_node(fragment.selection_set)
            self._recursively_referenced_fragments[operation] = fragments
        return fragments