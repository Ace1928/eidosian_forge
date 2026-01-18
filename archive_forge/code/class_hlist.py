from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class hlist(nodes.Element):
    """Node for "horizontal lists", i.e. lists that should be compressed to
    take up less vertical space.
    """