import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
@staticmethod
def _encode_status(status):
    """Cast status to bytes representation of current Python version.

        According to :pep:`3333`, when using Python 3, the response status
        and headers must be bytes masquerading as Unicode; that is, they
        must be of type "str" but are restricted to code points in the
        "Latin-1" set.
        """
    if not isinstance(status, str):
        raise TypeError('WSGI response status is not of type str.')
    return status.encode('ISO-8859-1')