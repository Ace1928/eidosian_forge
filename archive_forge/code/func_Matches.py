from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def Matches(self, path, is_dir=False):
    """Returns a Match for this pattern and the given path."""
    if self.must_be_dir and (not is_dir):
        return False
    if self._MatchesHelper(self.pattern.split('/'), path):
        return True
    else:
        return False