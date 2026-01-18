from __future__ import absolute_import, division, print_function
import os
def _default_getitems(*args):
    return dict(((key, self.get_option(key)) for key in args))