from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_punctuation(desc_sig_element):
    """Node for punctuation in a signature."""
    classes = ['p']