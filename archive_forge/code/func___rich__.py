import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def __rich__(self, parent: Optional['rich.tree.Tree']=None) -> 'rich.tree.Tree':
    """Returns a tree widget for the 'rich' library.

        Example:
            ::
                from rich import print
                from lark import Tree

                tree = Tree('root', ['node1', 'node2'])
                print(tree)
        """
    return self._rich(parent)