from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, cast
from ..error import GraphQLError
from ..language import (
from ..type import (
from ..utilities import TypeInfo, TypeInfoVisitor
class SDLValidationContext(ASTValidationContext):
    """Utility class providing a context for validation of an SDL AST.

    An instance of this class is passed as the context attribute to all Validators,
    allowing access to commonly useful contextual information from within a validation
    rule.
    """
    schema: Optional[GraphQLSchema]

    def __init__(self, ast: DocumentNode, schema: Optional[GraphQLSchema], on_error: Callable[[GraphQLError], None]) -> None:
        super().__init__(ast, on_error)
        self.schema = schema