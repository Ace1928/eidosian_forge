from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLInputField:
    """Definition of a GraphQL input field"""
    type: 'GraphQLInputType'
    default_value: Any
    description: Optional[str]
    deprecation_reason: Optional[str]
    out_name: Optional[str]
    extensions: Dict[str, Any]
    ast_node: Optional[InputValueDefinitionNode]

    def __init__(self, type_: 'GraphQLInputType', default_value: Any=Undefined, description: Optional[str]=None, deprecation_reason: Optional[str]=None, out_name: Optional[str]=None, extensions: Optional[Dict[str, Any]]=None, ast_node: Optional[InputValueDefinitionNode]=None) -> None:
        if not is_input_type(type_):
            raise TypeError('Input field type must be a GraphQL input type.')
        if description is not None and (not is_description(description)):
            raise TypeError('Input field description must be a string.')
        if deprecation_reason is not None and (not is_description(deprecation_reason)):
            raise TypeError('Input field deprecation reason must be a string.')
        if out_name is not None and (not isinstance(out_name, str)):
            raise TypeError('Input field out name must be a string.')
        if extensions is None:
            extensions = {}
        elif not isinstance(extensions, dict) or not all((isinstance(key, str) for key in extensions)):
            raise TypeError('Input field extensions must be a dictionary with string keys.')
        if ast_node and (not isinstance(ast_node, InputValueDefinitionNode)):
            raise TypeError('Input field AST node must be an InputValueDefinitionNode.')
        self.type = type_
        self.default_value = default_value
        self.description = description
        self.deprecation_reason = deprecation_reason
        self.out_name = out_name
        self.extensions = extensions
        self.ast_node = ast_node

    def __eq__(self, other: Any) -> bool:
        return self is other or (isinstance(other, GraphQLInputField) and self.type == other.type and (self.default_value == other.default_value) and (self.description == other.description) and (self.deprecation_reason == other.deprecation_reason) and (self.extensions == other.extensions) and (self.out_name == other.out_name))

    def to_kwargs(self) -> GraphQLInputFieldKwargs:
        return GraphQLInputFieldKwargs(type_=self.type, default_value=self.default_value, description=self.description, deprecation_reason=self.deprecation_reason, out_name=self.out_name, extensions=self.extensions, ast_node=self.ast_node)

    def __copy__(self) -> 'GraphQLInputField':
        return self.__class__(**self.to_kwargs())