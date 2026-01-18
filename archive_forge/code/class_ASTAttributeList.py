import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
class ASTAttributeList(ASTBaseBase):

    def __init__(self, attrs: List[ASTAttribute]) -> None:
        self.attrs = attrs

    def __len__(self) -> int:
        return len(self.attrs)

    def __add__(self, other: 'ASTAttributeList') -> 'ASTAttributeList':
        return ASTAttributeList(self.attrs + other.attrs)

    def _stringify(self, transform: StringifyTransform) -> str:
        return ' '.join((transform(attr) for attr in self.attrs))

    def describe_signature(self, signode: TextElement) -> None:
        if len(self.attrs) == 0:
            return
        self.attrs[0].describe_signature(signode)
        if len(self.attrs) == 1:
            return
        for attr in self.attrs[1:]:
            signode.append(addnodes.desc_sig_space())
            attr.describe_signature(signode)