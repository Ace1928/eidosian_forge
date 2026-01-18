import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _formatNetstring(data):
    return b''.join([str(len(data)).encode('ascii'), b':', data, b','])