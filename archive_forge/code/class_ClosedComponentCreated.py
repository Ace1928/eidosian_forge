import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
class ClosedComponentCreated(Exception):
    """
    We never want to create closed link components; the final answer
    is derived from when one has a created the "long knot" string
    link.
    """