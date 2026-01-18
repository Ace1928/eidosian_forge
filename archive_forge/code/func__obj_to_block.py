from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def _obj_to_block(objects, visited=None):
    items = list()
    for o in objects:
        if o not in items:
            items.append(o)
            for child in o._children:
                if child not in items:
                    items.append(child)
    return _obj_to_raw(items)