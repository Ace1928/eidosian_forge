import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _add_item(self, func, args):
    self._current_section.items.append((func, args))