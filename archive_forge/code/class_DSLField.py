from collections.abc import Iterable
import decimal
from functools import partial
from wandb_graphql.language import ast
from wandb_graphql.language.printer import print_ast
from wandb_graphql.type import (GraphQLField, GraphQLList,
from .utils import to_camel_case
class DSLField(object):

    def __init__(self, name, field):
        self.field = field
        self.ast_field = ast.Field(name=ast.Name(value=name), arguments=[])
        self.selection_set = None

    def select(self, *fields):
        if not self.ast_field.selection_set:
            self.ast_field.selection_set = ast.SelectionSet(selections=[])
        self.ast_field.selection_set.selections.extend(selections(*fields))
        return self

    def __call__(self, *args, **kwargs):
        return self.args(*args, **kwargs)

    def alias(self, alias):
        self.ast_field.alias = ast.Name(value=alias)
        return self

    def args(self, **args):
        for name, value in args.items():
            arg = self.field.args.get(name)
            arg_type_serializer = get_arg_serializer(arg.type)
            value = arg_type_serializer(value)
            self.ast_field.arguments.append(ast.Argument(name=ast.Name(value=name), value=get_ast_value(value)))
        return self

    @property
    def ast(self):
        return self.ast_field

    def __str__(self):
        return print_ast(self.ast_field)