from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class manpage(nodes.Inline, nodes.FixedTextElement):
    """Node for references to manpages."""