from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common._utils import get_all_subclasses
class LibMgr(PkgMgr):
    LIB = None

    def __init__(self):
        self._lib = None
        super(LibMgr, self).__init__()

    def is_available(self):
        found = False
        try:
            self._lib = __import__(self.LIB)
            found = True
        except ImportError:
            pass
        return found