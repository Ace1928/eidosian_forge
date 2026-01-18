import re
from ._exceptions import ProgrammingError
def _fetch_row(self, size=1):
    if not self._result:
        return ()
    return self._result.fetch_row(size, self._fetch_type)