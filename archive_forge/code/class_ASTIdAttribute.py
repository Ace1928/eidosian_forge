import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
class ASTIdAttribute(ASTAttribute):
    """For simple attributes defined by the user."""

    def __init__(self, id: str) -> None:
        self.id = id

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.id

    def describe_signature(self, signode: TextElement) -> None:
        signode.append(nodes.Text(self.id))