from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _more_generic(type1, type2):
    types = {type(None): 0, bool: 1, int: 2, float: 3, bytes: 4, str: 5}
    invtypes = {5: str, 4: bytes, 3: float, 2: int, 1: bool, 0: type(None)}
    moregeneric = max(types.get(type1, 5), types.get(type2, 5))
    return invtypes[moregeneric]