from ...error import GraphQLError
from ...language.visitor import Visitor
from ..validation_context import (
class SDLValidationRule(ASTValidationRule):
    """Visitor for validation of an SDL AST."""
    context: SDLValidationContext

    def __init__(self, context: SDLValidationContext) -> None:
        super().__init__(context)