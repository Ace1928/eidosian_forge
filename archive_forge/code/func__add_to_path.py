from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@staticmethod
def _add_to_path(path=None, additions=None):
    if path is None:
        path = ''
    if additions is None:
        additions = []
    for addition in additions:
        if path == '':
            path = '{}'.format(addition)
        else:
            path = '{}[{}]'.format(path, addition)
    return path