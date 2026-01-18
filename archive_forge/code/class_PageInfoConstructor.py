from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
class PageInfoConstructor(Protocol):

    def __call__(self, *, startCursor: Optional[ConnectionCursor], endCursor: Optional[ConnectionCursor], hasPreviousPage: bool, hasNextPage: bool) -> PageInfoType:
        ...