from sys import exc_info
from typing import Any, Collection, Dict, List, Optional, Union, TYPE_CHECKING
class GraphQLFormattedError(TypedDict, total=False):
    """Formatted GraphQL error"""
    message: str
    locations: List['FormattedSourceLocation']
    path: List[Union[str, int]]
    extensions: GraphQLErrorExtensions