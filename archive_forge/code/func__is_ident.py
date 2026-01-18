from __future__ import (absolute_import, division, print_function)
def _is_ident(cur):
    return cur > ' ' and cur not in ('"', '=')