import gyp.common
import json
import os
import posixpath
def _supplied_target_names_no_all(self):
    """Returns the supplied test targets without 'all'."""
    result = self._supplied_target_names()
    result.discard('all')
    return result