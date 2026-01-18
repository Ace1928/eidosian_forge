import re
import sys
from pprint import pprint
def _list_handle(self, listmatch):
    """Take apart a ``keyword=list('val, 'val')`` type string."""
    out = []
    name = listmatch.group(1)
    args = listmatch.group(2)
    for arg in self._list_members.findall(args):
        out.append(self._unquote(arg))
    return (name, out)