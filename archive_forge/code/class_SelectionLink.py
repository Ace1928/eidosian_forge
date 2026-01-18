import weakref
from collections import defaultdict
import param
from ..core.util import dimension_sanitizer
class SelectionLink(Link):
    """
    Links the selection between two glyph renderers.
    """
    _requires_target = True