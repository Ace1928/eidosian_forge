from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class not_smartquotable:
    """A node which does not support smart-quotes."""
    support_smartquotes = False