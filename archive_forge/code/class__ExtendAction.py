import os as _os
import re as _re
import sys as _sys
import warnings
from gettext import gettext as _, ngettext
class _ExtendAction(_AppendAction):

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.extend(values)
        setattr(namespace, self.dest, items)