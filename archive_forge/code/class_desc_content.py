from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_content(nodes.General, nodes.Element):
    """Node for object description content.

    Must be the last child node in a :py:class:`desc` node.
    """