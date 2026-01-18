from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc(nodes.Admonition, nodes.Element):
    """Node for a list of object signatures and a common description of them.

    Contains one or more :py:class:`desc_signature` nodes
    and then a single :py:class:`desc_content` node.

    This node always has two classes:

    - The name of the domain it belongs to, e.g., ``py`` or ``cpp``.
    - The name of the object type in the domain, e.g., ``function``.
    """