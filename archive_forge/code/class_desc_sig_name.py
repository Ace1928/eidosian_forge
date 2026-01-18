from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_name(desc_sig_element):
    """Node for an identifier in a signature."""
    classes = ['n']