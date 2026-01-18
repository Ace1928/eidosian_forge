import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
class SlottedTree(Tree):
    __slots__ = ('data', 'children', 'rule', '_meta')