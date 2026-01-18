import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _pop_action_class(self, kwargs, default=None):
    action = kwargs.pop('action', default)
    return self._registry_get('action', action, action)