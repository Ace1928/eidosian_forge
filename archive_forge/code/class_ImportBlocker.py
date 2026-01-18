import sys
import pytest
class ImportBlocker(object):
    """
    Block Imports

    To be placed on ``sys.meta_path``. This ensures that the modules
    specified cannot be imported, even if they are a builtin.
    """

    def __init__(self, *namestoblock):
        self.namestoblock = namestoblock

    def find_module(self, fullname, path=None):
        if fullname in self.namestoblock:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError('import of {0} is blocked'.format(fullname))