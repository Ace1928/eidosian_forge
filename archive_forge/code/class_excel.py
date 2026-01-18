import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
class excel(Dialect):
    """Describe the usual properties of Excel-generated CSV files."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = QUOTE_MINIMAL