import re
import sys
from pprint import pprint
def _handle_none(self, value):
    if value == 'None':
        return None
    elif value in ("'None'", '"None"'):
        value = self._unquote(value)
    return value