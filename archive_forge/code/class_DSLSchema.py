from collections.abc import Iterable
import decimal
from functools import partial
from wandb_graphql.language import ast
from wandb_graphql.language.printer import print_ast
from wandb_graphql.type import (GraphQLField, GraphQLList,
from .utils import to_camel_case
class DSLSchema(object):

    def __init__(self, client):
        self.client = client

    @property
    def schema(self):
        return self.client.schema

    def __getattr__(self, name):
        type_def = self.schema.get_type(name)
        return DSLType(type_def)

    def query(self, *args, **kwargs):
        return self.execute(query(*args, **kwargs))

    def mutate(self, *args, **kwargs):
        return self.query(*args, operation='mutate', **kwargs)

    def execute(self, document):
        return self.client.execute(document)