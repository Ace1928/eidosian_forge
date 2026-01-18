from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class pending_xref(nodes.Inline, nodes.Element):
    """Node for cross-references that cannot be resolved without complete
    information about all documents.

    These nodes are resolved before writing output, in
    BuildEnvironment.resolve_references.
    """
    child_text_separator = ''