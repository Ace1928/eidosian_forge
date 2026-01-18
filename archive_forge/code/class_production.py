from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class production(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for a single grammar production rule."""