import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _iter_indented_subactions(self, action):
    try:
        get_subactions = action._get_subactions
    except AttributeError:
        pass
    else:
        self._indent()
        for subaction in get_subactions():
            yield subaction
        self._dedent()