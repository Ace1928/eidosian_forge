import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _callable(obj):
    return hasattr(obj, '__call__') or hasattr(obj, '__bases__')