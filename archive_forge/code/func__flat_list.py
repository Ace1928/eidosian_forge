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
def _flat_list(nested_list):
    ret = []
    for item in nested_list:
        if isinstance(item, list):
            for subitem in item:
                ret.append(subitem)
        else:
            ret.append(item)
    return ret