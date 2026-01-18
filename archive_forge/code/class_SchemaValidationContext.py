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
class SchemaValidationContext:
    """Utility class providing a context for schema validation."""
    errors: List[GraphQLError]
    schema: GraphQLSchema

    def __init__(self, schema: GraphQLSchema):
        self.errors = []
        self.schema = schema

    def report_error(self, message: str, nodes: Union[Optional[Node], Collection[Optional[Node]]]=None) -> None:
        if nodes and (not isinstance(nodes, Node)):
            nodes = [node for node in nodes if node]
        nodes = cast(Optional[Collection[Node]], nodes)
        self.errors.append(GraphQLError(message, nodes))

    def validate_root_types(self) -> None:
        schema = self.schema
        query_type = schema.query_type
        if not query_type:
            self.report_error('Query root type must be provided.', schema.ast_node)
        elif not is_object_type(query_type):
            self.report_error(f'Query root type must be Object type, it cannot be {query_type}.', get_operation_type_node(schema, OperationType.QUERY) or query_type.ast_node)
        mutation_type = schema.mutation_type
        if mutation_type and (not is_object_type(mutation_type)):
            self.report_error(f'Mutation root type must be Object type if provided, it cannot be {mutation_type}.', get_operation_type_node(schema, OperationType.MUTATION) or mutation_type.ast_node)
        subscription_type = schema.subscription_type
        if subscription_type and (not is_object_type(subscription_type)):
            self.report_error(f'Subscription root type must be Object type if provided, it cannot be {subscription_type}.', get_operation_type_node(schema, OperationType.SUBSCRIPTION) or subscription_type.ast_node)

    def validate_directives(self) -> None:
        directives = self.schema.directives
        for directive in directives:
            if not is_directive(directive):
                self.report_error(f'Expected directive but got: {inspect(directive)}.', getattr(directive, 'ast_node', None))
                continue
            self.validate_name(directive)
            for arg_name, arg in directive.args.items():
                self.validate_name(arg, arg_name)
                if not is_input_type(arg.type):
                    self.report_error(f'The type of @{directive.name}({arg_name}:) must be Input Type but got: {inspect(arg.type)}.', arg.ast_node)
                if is_required_argument(arg) and arg.deprecation_reason is not None:
                    self.report_error(f'Required argument @{directive.name}({arg_name}:) cannot be deprecated.', [get_deprecated_directive_node(arg.ast_node), arg.ast_node and arg.ast_node.type])

    def validate_name(self, node: Any, name: Optional[str]=None) -> None:
        try:
            if not name:
                name = node.name
            name = cast(str, name)
            ast_node = node.ast_node
        except AttributeError:
            pass
        else:
            if name.startswith('__'):
                self.report_error(f"Name {name!r} must not begin with '__', which is reserved by GraphQL introspection.", ast_node)

    def validate_types(self) -> None:
        validate_input_object_circular_refs = InputObjectCircularRefsValidator(self)
        for type_ in self.schema.type_map.values():
            if not is_named_type(type_):
                self.report_error(f'Expected GraphQL named type but got: {inspect(type_)}.', type_.ast_node if is_named_type(type_) else None)
                continue
            if not is_introspection_type(type_):
                self.validate_name(type_)
            if is_object_type(type_):
                type_ = cast(GraphQLObjectType, type_)
                self.validate_fields(type_)
                self.validate_interfaces(type_)
            elif is_interface_type(type_):
                type_ = cast(GraphQLInterfaceType, type_)
                self.validate_fields(type_)
                self.validate_interfaces(type_)
            elif is_union_type(type_):
                type_ = cast(GraphQLUnionType, type_)
                self.validate_union_members(type_)
            elif is_enum_type(type_):
                type_ = cast(GraphQLEnumType, type_)
                self.validate_enum_values(type_)
            elif is_input_object_type(type_):
                type_ = cast(GraphQLInputObjectType, type_)
                self.validate_input_fields(type_)
                validate_input_object_circular_refs(type_)

    def validate_fields(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> None:
        fields = type_.fields
        if not fields:
            self.report_error(f'Type {type_.name} must define one or more fields.', [type_.ast_node, *type_.extension_ast_nodes])
        for field_name, field in fields.items():
            self.validate_name(field, field_name)
            if not is_output_type(field.type):
                self.report_error(f'The type of {type_.name}.{field_name} must be Output Type but got: {inspect(field.type)}.', field.ast_node and field.ast_node.type)
            for arg_name, arg in field.args.items():
                self.validate_name(arg, arg_name)
                if not is_input_type(arg.type):
                    self.report_error(f'The type of {type_.name}.{field_name}({arg_name}:) must be Input Type but got: {inspect(arg.type)}.', arg.ast_node and arg.ast_node.type)
                if is_required_argument(arg) and arg.deprecation_reason is not None:
                    self.report_error(f'Required argument {type_.name}.{field_name}({arg_name}:) cannot be deprecated.', [get_deprecated_directive_node(arg.ast_node), arg.ast_node and arg.ast_node.type])

    def validate_interfaces(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> None:
        iface_type_names: Set[str] = set()
        for iface in type_.interfaces:
            if not is_interface_type(iface):
                self.report_error(f'Type {type_.name} must only implement Interface types, it cannot implement {inspect(iface)}.', get_all_implements_interface_nodes(type_, iface))
                continue
            if type_ is iface:
                self.report_error(f'Type {type_.name} cannot implement itself because it would create a circular reference.', get_all_implements_interface_nodes(type_, iface))
            if iface.name in iface_type_names:
                self.report_error(f'Type {type_.name} can only implement {iface.name} once.', get_all_implements_interface_nodes(type_, iface))
                continue
            iface_type_names.add(iface.name)
            self.validate_type_implements_ancestors(type_, iface)
            self.validate_type_implements_interface(type_, iface)

    def validate_type_implements_interface(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType], iface: GraphQLInterfaceType) -> None:
        type_fields, iface_fields = (type_.fields, iface.fields)
        for field_name, iface_field in iface_fields.items():
            type_field = type_fields.get(field_name)
            if not type_field:
                self.report_error(f'Interface field {iface.name}.{field_name} expected but {type_.name} does not provide it.', [iface_field.ast_node, type_.ast_node, *type_.extension_ast_nodes])
                continue
            if not is_type_sub_type_of(self.schema, type_field.type, iface_field.type):
                self.report_error(f'Interface field {iface.name}.{field_name} expects type {iface_field.type} but {type_.name}.{field_name} is type {type_field.type}.', [iface_field.ast_node and iface_field.ast_node.type, type_field.ast_node and type_field.ast_node.type])
            for arg_name, iface_arg in iface_field.args.items():
                type_arg = type_field.args.get(arg_name)
                if not type_arg:
                    self.report_error(f'Interface field argument {iface.name}.{field_name}({arg_name}:) expected but {type_.name}.{field_name} does not provide it.', [iface_arg.ast_node, type_field.ast_node])
                    continue
                if not is_equal_type(iface_arg.type, type_arg.type):
                    self.report_error(f'Interface field argument {iface.name}.{field_name}({arg_name}:) expects type {iface_arg.type} but {type_.name}.{field_name}({arg_name}:) is type {type_arg.type}.', [iface_arg.ast_node and iface_arg.ast_node.type, type_arg.ast_node and type_arg.ast_node.type])
            for arg_name, type_arg in type_field.args.items():
                iface_arg = iface_field.args.get(arg_name)
                if not iface_arg and is_required_argument(type_arg):
                    self.report_error(f'Object field {type_.name}.{field_name} includes required argument {arg_name} that is missing from the Interface field {iface.name}.{field_name}.', [type_arg.ast_node, iface_field.ast_node])

    def validate_type_implements_ancestors(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType], iface: GraphQLInterfaceType) -> None:
        type_interfaces, iface_interfaces = (type_.interfaces, iface.interfaces)
        for transitive in iface_interfaces:
            if transitive not in type_interfaces:
                self.report_error(f'Type {type_.name} cannot implement {iface.name} because it would create a circular reference.' if transitive is type_ else f'Type {type_.name} must implement {transitive.name} because it is implemented by {iface.name}.', get_all_implements_interface_nodes(iface, transitive) + get_all_implements_interface_nodes(type_, iface))

    def validate_union_members(self, union: GraphQLUnionType) -> None:
        member_types = union.types
        if not member_types:
            self.report_error(f'Union type {union.name} must define one or more member types.', [union.ast_node, *union.extension_ast_nodes])
        included_type_names: Set[str] = set()
        for member_type in member_types:
            if is_object_type(member_type):
                if member_type.name in included_type_names:
                    self.report_error(f'Union type {union.name} can only include type {member_type.name} once.', get_union_member_type_nodes(union, member_type.name))
                else:
                    included_type_names.add(member_type.name)
            else:
                self.report_error(f'Union type {union.name} can only include Object types, it cannot include {inspect(member_type)}.', get_union_member_type_nodes(union, str(member_type)))

    def validate_enum_values(self, enum_type: GraphQLEnumType) -> None:
        enum_values = enum_type.values
        if not enum_values:
            self.report_error(f'Enum type {enum_type.name} must define one or more values.', [enum_type.ast_node, *enum_type.extension_ast_nodes])
        for value_name, enum_value in enum_values.items():
            self.validate_name(enum_value, value_name)

    def validate_input_fields(self, input_obj: GraphQLInputObjectType) -> None:
        fields = input_obj.fields
        if not fields:
            self.report_error(f'Input Object type {input_obj.name} must define one or more fields.', [input_obj.ast_node, *input_obj.extension_ast_nodes])
        for field_name, field in fields.items():
            self.validate_name(field, field_name)
            if not is_input_type(field.type):
                self.report_error(f'The type of {input_obj.name}.{field_name} must be Input Type but got: {inspect(field.type)}.', field.ast_node.type if field.ast_node else None)
            if is_required_input_field(field) and field.deprecation_reason is not None:
                self.report_error(f'Required input field {input_obj.name}.{field_name} cannot be deprecated.', [get_deprecated_directive_node(field.ast_node), field.ast_node and field.ast_node.type])