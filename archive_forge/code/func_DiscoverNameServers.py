import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def DiscoverNameServers():
    """Don't call, only here for backward compatability.  We do discovery for
    you automatically.
    """
    pass