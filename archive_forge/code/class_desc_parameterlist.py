from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_parameterlist(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for a general parameter list."""
    child_text_separator = ', '

    def astext(self):
        return '({})'.format(super().astext())