from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
class PageInfoType(Protocol):

    @property
    def startCursor(self) -> Optional[ConnectionCursor]:
        ...

    def endCursor(self) -> Optional[ConnectionCursor]:
        ...

    def hasPreviousPage(self) -> bool:
        ...

    def hasNextPage(self) -> bool:
        ...