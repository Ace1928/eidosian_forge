import os
from breezy import tests
def _make_versioned_file(self, path, line_prefix='line', total_lines=10):
    self._make_file(path, line_prefix, total_lines, versioned=True)