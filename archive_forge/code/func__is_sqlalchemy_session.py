from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _is_sqlalchemy_session(dbo):
    return _hasmethods(dbo, 'execute', 'connection', 'get_bind')