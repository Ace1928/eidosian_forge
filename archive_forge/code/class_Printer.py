import logging
from pyasn1 import __version__
from pyasn1 import error
from pyasn1.compat.octets import octs2ints
class Printer(object):

    def __init__(self, logger=None, handler=None, formatter=None):
        if logger is None:
            logger = logging.getLogger('pyasn1')
        logger.setLevel(logging.DEBUG)
        if handler is None:
            handler = logging.StreamHandler()
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.__logger = logger

    def __call__(self, msg):
        self.__logger.debug(msg)

    def __str__(self):
        return '<python logging>'