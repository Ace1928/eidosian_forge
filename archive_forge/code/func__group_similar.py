import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def _group_similar(items: List[T], comparer: Callable[[T, T], bool]) -> List[List[T]]:
    """Combines similar items into groups.

    Args:
      items: The list of items to group.
      comparer: Determines if two items are similar.

    Returns:
      A list of groups of items.
    """
    groups: List[List[T]] = []
    used: Set[int] = set()
    for i in range(len(items)):
        if i not in used:
            group = [items[i]]
            for j in range(i + 1, len(items)):
                if j not in used and comparer(items[i], items[j]):
                    used.add(j)
                    group.append(items[j])
            groups.append(group)
    return groups