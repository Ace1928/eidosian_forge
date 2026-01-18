from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class productionlist(nodes.Admonition, nodes.Element):
    """Node for grammar production lists.

    Contains ``production`` nodes.
    """