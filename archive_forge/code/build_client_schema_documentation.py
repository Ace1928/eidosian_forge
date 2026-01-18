from itertools import chain
from typing import cast, Callable, Collection, Dict, List, Union
from ..language import DirectiveLocation, parse_value
from ..pyutils import inspect, Undefined
from ..type import (
from .get_introspection_query import (
from .value_from_ast import value_from_ast
Build a GraphQLSchema for use by client tools.

    Given the result of a client running the introspection query, creates and returns
    a GraphQLSchema instance which can be then used with all GraphQL-core 3 tools,
    but cannot be used to execute a query, as introspection does not represent the
    "resolver", "parse" or "serialize" functions or any other server-internal
    mechanisms.

    This function expects a complete introspection result. Don't forget to check the
    "errors" field of a server response before calling this function.
    