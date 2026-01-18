import logging
from pyasn1 import __version__
from pyasn1 import error
from pyasn1.compat.octets import octs2ints
class NullHandler(logging.Handler):

    def emit(self, record):
        pass