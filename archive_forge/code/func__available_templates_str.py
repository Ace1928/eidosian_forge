import textwrap
import pkgutil
import copy
import os
import json
from functools import reduce
def _available_templates_str(self):
    """
        Return nicely wrapped string representation of all
        available template names
        """
    available = '\n'.join(textwrap.wrap(repr(list(self)), width=79 - 8, initial_indent=' ' * 8, subsequent_indent=' ' * 9))
    return available