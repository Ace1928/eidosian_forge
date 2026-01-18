from __future__ import absolute_import, division, print_function
import os
def _default_default_setter(key, default=None):
    try:
        value = self.get_option(key)
        return value
    except KeyError:
        self.set_option(key, default)
        return default