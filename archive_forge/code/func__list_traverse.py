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
def _list_traverse(self, as_edge: bool=False, *args: Any, **kwargs: Any) -> IterableList[Union['Commit', 'Submodule', 'Tree', 'Blob']]:
    """Traverse self and collect all items found.

        :return: IterableList with the results of the traversal as produced by
            :meth:`traverse`::

                Commit -> IterableList['Commit']
                Submodule ->  IterableList['Submodule']
                Tree -> IterableList[Union['Submodule', 'Tree', 'Blob']]
        """
    if isinstance(self, Has_id_attribute):
        id = self._id_attribute_
    else:
        id = ''
    if not as_edge:
        out: IterableList[Union['Commit', 'Submodule', 'Tree', 'Blob']] = IterableList(id)
        out.extend(self.traverse(*args, as_edge=as_edge, **kwargs))
        return out
    else:
        out_list: IterableList = IterableList(self.traverse(*args, **kwargs))
        return out_list