from operator import attrgetter, itemgetter
from typing import (
from ..error import GraphQLError
from ..pyutils import inspect
from ..language import (
from .definition import (
from ..utilities.type_comparators import is_equal_type, is_type_sub_type_of
from .directives import is_directive, GraphQLDeprecatedDirective
from .introspection import is_introspection_type
from .schema import GraphQLSchema, assert_schema
class InputObjectCircularRefsValidator:
    """Modified copy of algorithm from validation.rules.NoFragmentCycles"""

    def __init__(self, context: SchemaValidationContext):
        self.context = context
        self.visited_types: Set[str] = set()
        self.field_path: List[Tuple[str, GraphQLInputField]] = []
        self.field_path_index_by_type_name: Dict[str, int] = {}

    def __call__(self, input_obj: GraphQLInputObjectType) -> None:
        """Detect cycles recursively."""
        name = input_obj.name
        if name in self.visited_types:
            return
        self.visited_types.add(name)
        self.field_path_index_by_type_name[name] = len(self.field_path)
        for field_name, field in input_obj.fields.items():
            if is_non_null_type(field.type) and is_input_object_type(field.type.of_type):
                field_type = cast(GraphQLInputObjectType, field.type.of_type)
                cycle_index = self.field_path_index_by_type_name.get(field_type.name)
                self.field_path.append((field_name, field))
                if cycle_index is None:
                    self(field_type)
                else:
                    cycle_path = self.field_path[cycle_index:]
                    field_names = map(itemgetter(0), cycle_path)
                    self.context.report_error(f"Cannot reference Input Object '{field_type.name}' within itself through a series of non-null fields: '{'.'.join(field_names)}'.", cast(Collection[Node], map(attrgetter('ast_node'), map(itemgetter(1), cycle_path))))
                self.field_path.pop()
        del self.field_path_index_by_type_name[name]