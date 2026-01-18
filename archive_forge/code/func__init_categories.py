import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _init_categories(categories=categories):
    for k, v in globals().items():
        if k[:3] == 'LC_':
            categories[k] = v