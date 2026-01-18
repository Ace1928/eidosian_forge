from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_name(_desc_classes_injector, nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for the main object name.

    For example, in the declaration of a Python class ``MyModule.MyClass``,
    the main name is ``MyClass``.

    This node always has the class ``sig-name``.
    """
    classes = ['sig-name', 'descname']