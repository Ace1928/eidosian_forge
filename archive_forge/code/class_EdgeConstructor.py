from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
class EdgeConstructor(Protocol):

    def __call__(self, *, node: Any, cursor: ConnectionCursor) -> EdgeType:
        ...