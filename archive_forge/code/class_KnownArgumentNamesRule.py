from typing import cast, Any, Dict, List, Union
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list
from ...type import specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
class KnownArgumentNamesRule(KnownArgumentNamesOnDirectivesRule):
    """Known argument names

    A GraphQL field is only valid if all supplied arguments are defined by that field.

    See https://spec.graphql.org/draft/#sec-Argument-Names
    See https://spec.graphql.org/draft/#sec-Directives-Are-In-Valid-Locations
    """
    context: ValidationContext

    def __init__(self, context: ValidationContext):
        super().__init__(context)

    def enter_argument(self, arg_node: ArgumentNode, *args: Any) -> None:
        context = self.context
        arg_def = context.get_argument()
        field_def = context.get_field_def()
        parent_type = context.get_parent_type()
        if not arg_def and field_def and parent_type:
            arg_name = arg_node.name.value
            field_name = args[3][-1].name.value
            known_args_names = list(field_def.args)
            suggestions = suggestion_list(arg_name, known_args_names)
            context.report_error(GraphQLError(f"Unknown argument '{arg_name}' on field '{parent_type.name}.{field_name}'." + did_you_mean(suggestions), arg_node))