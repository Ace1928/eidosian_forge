from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class download_reference(nodes.reference):
    """Node for download references, similar to pending_xref."""