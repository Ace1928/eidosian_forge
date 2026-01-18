from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.six import PY2
class FakeURLLIB3Connection(object):

    def __init__(self):
        self.HTTPConnection = _HTTPConnection