from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
def _c2index_isreverse(c1, c2):
    """
    Private helper function to get the index c2_0_index of the first node of cycle c1
    in cycle c2 and whether the cycle c2 should be reversed or not.

    Returns None if the first node of cycle c1 is not found in cycle c2.
    The reverse value depends on the index c2_1_index of the second node of cycle c1 in
    cycle c2 : if it is *just after* the c2_0_index, reverse is False, if it is
    *just before* the c2_0_index, reverse is True, otherwise the function returns None).
    """
    c1_0 = c1.nodes[0]
    c1_1 = c1.nodes[1]
    if c1_0 not in c2.nodes:
        return (None, None, 'First node of cycle c1 not found in cycle c2.')
    if c1_1 not in c2.nodes:
        return (None, None, 'Second node of cycle c1 not found in cycle c2.')
    c2_0_index = c2.nodes.index(c1_0)
    c2_1_index = c2.nodes.index(c1_1)
    if c2_0_index == 0:
        if c2_1_index == 1:
            reverse = False
        elif c2_1_index == len(c2.nodes) - 1:
            reverse = True
        else:
            msg = 'Second node of cycle c1 is not second or last in cycle c2 (first node of cycle c1 is first in cycle c2).'
            return (None, None, msg)
    elif c2_0_index == len(c2.nodes) - 1:
        if c2_1_index == 0:
            reverse = False
        elif c2_1_index == c2_0_index - 1:
            reverse = True
        else:
            msg = 'Second node of cycle c1 is not first or before last in cycle c2 (first node of cycle c1 is last in cycle c2).'
            return (None, None, msg)
    elif c2_1_index == c2_0_index + 1:
        reverse = False
    elif c2_1_index == c2_0_index - 1:
        reverse = True
    else:
        msg = 'Second node of cycle c1 in cycle c2 is not just after or just before first node of cycle c1 in cycle c2.'
        return (None, None, msg)
    return (c2_0_index, reverse, '')