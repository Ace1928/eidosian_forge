from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common._utils import get_all_subclasses
class CLIMgr(PkgMgr):
    CLI = None

    def __init__(self):
        self._cli = None
        super(CLIMgr, self).__init__()

    def is_available(self):
        try:
            self._cli = get_bin_path(self.CLI)
        except ValueError:
            return False
        return True