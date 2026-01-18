from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLNamedType(GraphQLType):
    """Base class for all GraphQL named types"""
    name: str
    description: Optional[str]
    extensions: Dict[str, Any]
    ast_node: Optional[TypeDefinitionNode]
    extension_ast_nodes: Tuple[TypeExtensionNode, ...]

    def __init__(self, name: str, description: Optional[str]=None, extensions: Optional[Dict[str, Any]]=None, ast_node: Optional[TypeDefinitionNode]=None, extension_ast_nodes: Optional[Collection[TypeExtensionNode]]=None) -> None:
        assert_name(name)
        if description is not None and (not is_description(description)):
            raise TypeError('The description must be a string.')
        if extensions is None:
            extensions = {}
        elif not isinstance(extensions, dict) or not all((isinstance(key, str) for key in extensions)):
            raise TypeError(f'{name} extensions must be a dictionary with string keys.')
        if ast_node and (not isinstance(ast_node, TypeDefinitionNode)):
            raise TypeError(f'{name} AST node must be a TypeDefinitionNode.')
        if extension_ast_nodes:
            if not is_collection(extension_ast_nodes) or not all((isinstance(node, TypeExtensionNode) for node in extension_ast_nodes)):
                raise TypeError(f'{name} extension AST nodes must be specified as a collection of TypeExtensionNode instances.')
            if not isinstance(extension_ast_nodes, tuple):
                extension_ast_nodes = tuple(extension_ast_nodes)
        else:
            extension_ast_nodes = ()
        self.name = name
        self.description = description
        self.extensions = extensions
        self.ast_node = ast_node
        self.extension_ast_nodes = extension_ast_nodes

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r}>'

    def __str__(self) -> str:
        return self.name

    def to_kwargs(self) -> GraphQLNamedTypeKwargs:
        return GraphQLNamedTypeKwargs(name=self.name, description=self.description, extensions=self.extensions, ast_node=self.ast_node, extension_ast_nodes=self.extension_ast_nodes)

    def __copy__(self) -> 'GraphQLNamedType':
        return self.__class__(**self.to_kwargs())