import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
class FrozenNode(Exception):
    """Exception raised when a frozen node is modified."""

    def __init__(self):
        super(FrozenNode, self).__init__("Frozen node(s) can't be modified")