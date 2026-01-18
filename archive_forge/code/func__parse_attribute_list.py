import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def _parse_attribute_list(self) -> ASTAttributeList:
    res = []
    while True:
        attr = self._parse_attribute()
        if attr is None:
            break
        res.append(attr)
    return ASTAttributeList(res)