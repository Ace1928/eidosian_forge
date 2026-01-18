import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _convert_nodes_to_standalone_comment(nodes: Sequence[LN], *, newline: Leaf) -> None:
    """Convert nodes to STANDALONE_COMMENT by modifying the tree inline."""
    if not nodes:
        return
    parent = nodes[0].parent
    first = first_leaf(nodes[0])
    if not parent or not first:
        return
    prefix = first.prefix
    first.prefix = ''
    value = ''.join((str(node) for node in nodes))
    if newline.prefix:
        value += newline.prefix
        newline.prefix = ''
    index = nodes[0].remove()
    for node in nodes[1:]:
        node.remove()
    if index is not None:
        parent.insert_child(index, Leaf(STANDALONE_COMMENT, value, prefix=prefix, fmt_pass_converted_first_leaf=first))