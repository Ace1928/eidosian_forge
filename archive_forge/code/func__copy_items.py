import os as _os
import re as _re
import sys as _sys
import warnings
from gettext import gettext as _, ngettext
def _copy_items(items):
    if items is None:
        return []
    if type(items) is list:
        return items[:]
    import copy
    return copy.copy(items)