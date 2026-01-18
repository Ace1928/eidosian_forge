from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def _expand_block(self, configobj, S=None):
    if S is None:
        S = list()
    S.append(configobj)
    for child in configobj._children:
        if child in S:
            continue
        self._expand_block(child, S)
    return S