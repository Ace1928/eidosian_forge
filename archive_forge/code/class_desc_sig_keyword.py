from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_keyword(desc_sig_element):
    """Node for a general keyword in a signature."""
    classes = ['k']