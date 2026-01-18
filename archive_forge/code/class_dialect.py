import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
class dialect(Dialect):
    _name = 'sniffed'
    lineterminator = '\r\n'
    quoting = QUOTE_MINIMAL