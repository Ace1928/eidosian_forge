from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _is_sqlalchemy_engine(dbo):
    return _hasmethods(dbo, 'execute', 'connect', 'raw_connection') and _hasprop(dbo, 'driver')