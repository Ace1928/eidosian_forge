from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_literal_string(desc_sig_element):
    """Node for a string literal in a signature."""
    classes = ['s']