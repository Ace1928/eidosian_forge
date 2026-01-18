from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_type(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for return types or object type names."""