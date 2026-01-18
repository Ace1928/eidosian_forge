import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _registry_get(self, registry_name, value, default=None):
    return self._registries[registry_name].get(value, default)