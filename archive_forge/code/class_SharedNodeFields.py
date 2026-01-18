import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
class SharedNodeFields:
    shared = String()
    something_else = String()

    def resolve_something_else(*_):
        return '----'