import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
def __unicode__(self):
    try:
        return self._value.decode(self.encoding)
    except UnicodeDecodeError:
        exc = sys.exc_info()[1]
        raise error.PyAsn1UnicodeDecodeError("Can't decode string '%s' with codec %s" % (self._value, self.encoding), exc)