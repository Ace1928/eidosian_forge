from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def addToStack(stack: Deque[TraverseNT], src_item: 'Traversable', branch_first: bool, depth: int) -> None:
    lst = self._get_intermediate_items(item)
    if not lst:
        return
    if branch_first:
        stack.extendleft((TraverseNT(depth, i, src_item) for i in lst))
    else:
        reviter = (TraverseNT(depth, lst[i], src_item) for i in range(len(lst) - 1, -1, -1))
        stack.extend(reviter)