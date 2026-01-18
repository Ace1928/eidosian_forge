from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class highlightlang(nodes.Element):
    """Inserted to set the highlight language and line number options for
    subsequent code blocks.
    """