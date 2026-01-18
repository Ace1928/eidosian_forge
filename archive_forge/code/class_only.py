from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class only(nodes.Element):
    """Node for "only" directives (conditional inclusion based on tags)."""