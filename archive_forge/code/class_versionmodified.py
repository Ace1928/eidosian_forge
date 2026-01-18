from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class versionmodified(nodes.Admonition, nodes.TextElement):
    """Node for version change entries.

    Currently used for "versionadded", "versionchanged" and "deprecated"
    directives.
    """