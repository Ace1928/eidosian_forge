from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class literal_strong(nodes.strong, not_smartquotable):
    """Node that behaves like `strong`, but further text processors are not
    applied (e.g. smartypants for HTML output).
    """