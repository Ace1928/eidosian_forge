import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _find_one(self, method, name, attrs, string, **kwargs):
    r = None
    l = method(name, attrs, string, 1, _stacklevel=4, **kwargs)
    if l:
        r = l[0]
    return r