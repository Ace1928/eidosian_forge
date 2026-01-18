from typing import TYPE_CHECKING
from typing import List
from typing import Optional
class _PrintOpts(TypedDict, total=False):
    margin: _MarginOpts
    page: _PageOpts
    background: bool
    orientation: Orientation
    scale: float
    shrinkToFit: bool
    pageRanges: List[str]