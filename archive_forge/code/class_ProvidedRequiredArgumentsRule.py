from typing import cast, Any, Dict, List, Union
from ...error import GraphQLError
from ...language import (
from ...type import GraphQLArgument, is_required_argument, is_type, specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
class ProvidedRequiredArgumentsRule(ProvidedRequiredArgumentsOnDirectivesRule):
    """Provided required arguments

    A field or directive is only valid if all required (non-null without a default
    value) field arguments have been provided.
    """
    context: ValidationContext

    def __init__(self, context: ValidationContext):
        super().__init__(context)

    def leave_field(self, field_node: FieldNode, *_args: Any) -> VisitorAction:
        field_def = self.context.get_field_def()
        if not field_def:
            return SKIP
        arg_nodes = field_node.arguments or ()
        arg_node_map = {arg.name.value: arg for arg in arg_nodes}
        for arg_name, arg_def in field_def.args.items():
            arg_node = arg_node_map.get(arg_name)
            if not arg_node and is_required_argument(arg_def):
                self.report_error(GraphQLError(f"Field '{field_node.name.value}' argument '{arg_name}' of type '{arg_def.type}' is required, but it was not provided.", field_node))
        return None