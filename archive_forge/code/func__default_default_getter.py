from __future__ import absolute_import, division, print_function
import os
def _default_default_getter(key, default):
    try:
        return self.get_option(key)
    except KeyError:
        return default