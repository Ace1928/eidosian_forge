import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
class ASTCPPAttribute(ASTAttribute):

    def __init__(self, arg: str) -> None:
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        return '[[' + self.arg + ']]'

    def describe_signature(self, signode: TextElement) -> None:
        signode.append(addnodes.desc_sig_punctuation('[[', '[['))
        signode.append(nodes.Text(self.arg))
        signode.append(addnodes.desc_sig_punctuation(']]', ']]'))