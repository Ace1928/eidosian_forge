from .. import (
import stat
def _find_interesting_from(self, commit_ref):
    if commit_ref is None:
        return None
    return self._find_interesting_parent(commit_ref)