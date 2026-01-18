from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class _desc_classes_injector(nodes.Element, not_smartquotable):
    """Helper base class for injecting a fixed list of classes.

    Use as the first base class.
    """
    classes: List[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self['classes'].extend(self.classes)