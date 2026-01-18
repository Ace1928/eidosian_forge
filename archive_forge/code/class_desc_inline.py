from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_inline(_desc_classes_injector, nodes.Inline, nodes.TextElement):
    """Node for a signature fragment in inline text.

    This is for example used for roles like :rst:role:`cpp:expr`.

    This node always has the classes ``sig``, ``sig-inline``,
    and the name of the domain it belongs to.
    """
    classes = ['sig', 'sig-inline']

    def __init__(self, domain: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self['classes'].append(domain)