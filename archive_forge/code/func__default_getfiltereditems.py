from __future__ import absolute_import, division, print_function
import os
def _default_getfiltereditems(filter, *args):
    return dict(((key, value) for key, value in self.get_options(*args).items() if filter(key, value)))