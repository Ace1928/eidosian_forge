from textwrap import dedent
from pytest import raises
from graphql.type import GraphQLObjectType, GraphQLSchema
from ..field import Field
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
class MyOtherType(ObjectType):
    field = String()