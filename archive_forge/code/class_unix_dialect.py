import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
class unix_dialect(Dialect):
    """Describe the usual properties of Unix-generated CSV files."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = QUOTE_ALL