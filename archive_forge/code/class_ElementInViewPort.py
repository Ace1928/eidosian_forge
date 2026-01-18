import time
from sys import platform
from typing import (
class ElementInViewPort(TypedDict):
    """A typed dictionary containing information about elements in the viewport."""
    node_index: str
    backend_node_id: int
    node_name: Optional[str]
    node_value: Optional[str]
    node_meta: List[str]
    is_clickable: bool
    origin_x: int
    origin_y: int
    center_x: int
    center_y: int