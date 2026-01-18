from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_space(desc_sig_element):
    """Node for a space in a signature."""
    classes = ['w']

    def __init__(self, rawsource: str='', text: str=' ', *children: Element, **attributes: Any) -> None:
        super().__init__(rawsource, text, *children, **attributes)