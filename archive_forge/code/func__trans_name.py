from webob.compat import (
from webob.multidict import MultiDict
def _trans_name(name):
    name = name.upper()
    if name in header2key:
        return header2key[name]
    return 'HTTP_' + name.replace('-', '_')