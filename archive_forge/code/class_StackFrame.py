import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class StackFrame(NamedTuple):
    parent: Any
    node: Any
    indents: list
    this_islast: bool
    this_vertical: bool