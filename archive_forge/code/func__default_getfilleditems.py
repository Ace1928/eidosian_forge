from __future__ import absolute_import, division, print_function
import os
def _default_getfilleditems(*args):
    return self.get_filtered_options(lambda k, v: v is not None, *args)