from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_addname(_desc_classes_injector, nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for additional name parts for an object.

    For example, in the declaration of a Python class ``MyModule.MyClass``,
    the additional name part is ``MyModule.``.

    This node always has the class ``sig-prename``.
    """
    classes = ['sig-prename', 'descclassname']