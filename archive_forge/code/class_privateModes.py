from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
class privateModes:
    """
    ANSI-Compatible Private Modes
    """
    ERROR = 0
    CURSOR_KEY = 1
    ANSI_VT52 = 2
    COLUMN = 3
    SCROLL = 4
    SCREEN = 5
    ORIGIN = 6
    AUTO_WRAP = 7
    AUTO_REPEAT = 8
    PRINTER_FORM_FEED = 18
    PRINTER_EXTENT = 19
    CURSOR_MODE = 25