import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
class Token(NamedTuple):
    category: Pattern
    value: Any
    line: int
    position: int