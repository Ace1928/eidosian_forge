from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _is_sqlalchemy_connection(dbo):
    return _hasmethod(dbo, 'execute') and _hasprop(dbo, 'connection')