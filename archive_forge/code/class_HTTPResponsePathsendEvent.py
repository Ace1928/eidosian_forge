import sys
from typing import (
class HTTPResponsePathsendEvent(TypedDict):
    type: Literal['http.response.pathsend']
    path: str