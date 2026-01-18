import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
class Marshaller:
    """Generate an XML-RPC params chunk from a Python data structure.

    Create a Marshaller instance for each set of parameters, and use
    the "dumps" method to convert your data (represented as a tuple)
    to an XML-RPC params chunk.  To write a fault response, pass a
    Fault instance instead.  You may prefer to use the "dumps" module
    function for this purpose.
    """

    def __init__(self, encoding=None, allow_none=False):
        self.memo = {}
        self.data = None
        self.encoding = encoding
        self.allow_none = allow_none
    dispatch = {}

    def dumps(self, values):
        out = []
        write = out.append
        dump = self.__dump
        if isinstance(values, Fault):
            write('<fault>\n')
            dump({'faultCode': values.faultCode, 'faultString': values.faultString}, write)
            write('</fault>\n')
        else:
            write('<params>\n')
            for v in values:
                write('<param>\n')
                dump(v, write)
                write('</param>\n')
            write('</params>\n')
        result = ''.join(out)
        return result

    def __dump(self, value, write):
        try:
            f = self.dispatch[type(value)]
        except KeyError:
            if not hasattr(value, '__dict__'):
                raise TypeError('cannot marshal %s objects' % type(value))
            for type_ in type(value).__mro__:
                if type_ in self.dispatch.keys():
                    raise TypeError('cannot marshal %s objects' % type(value))
            f = self.dispatch['_arbitrary_instance']
        f(self, value, write)

    def dump_nil(self, value, write):
        if not self.allow_none:
            raise TypeError('cannot marshal None unless allow_none is enabled')
        write('<value><nil/></value>')
    dispatch[type(None)] = dump_nil

    def dump_bool(self, value, write):
        write('<value><boolean>')
        write(value and '1' or '0')
        write('</boolean></value>\n')
    dispatch[bool] = dump_bool

    def dump_long(self, value, write):
        if value > MAXINT or value < MININT:
            raise OverflowError('int exceeds XML-RPC limits')
        write('<value><int>')
        write(str(int(value)))
        write('</int></value>\n')
    dispatch[int] = dump_long
    dump_int = dump_long

    def dump_double(self, value, write):
        write('<value><double>')
        write(repr(value))
        write('</double></value>\n')
    dispatch[float] = dump_double

    def dump_unicode(self, value, write, escape=escape):
        write('<value><string>')
        write(escape(value))
        write('</string></value>\n')
    dispatch[str] = dump_unicode

    def dump_bytes(self, value, write):
        write('<value><base64>\n')
        encoded = base64.encodebytes(value)
        write(encoded.decode('ascii'))
        write('</base64></value>\n')
    dispatch[bytes] = dump_bytes
    dispatch[bytearray] = dump_bytes

    def dump_array(self, value, write):
        i = id(value)
        if i in self.memo:
            raise TypeError('cannot marshal recursive sequences')
        self.memo[i] = None
        dump = self.__dump
        write('<value><array><data>\n')
        for v in value:
            dump(v, write)
        write('</data></array></value>\n')
        del self.memo[i]
    dispatch[tuple] = dump_array
    dispatch[list] = dump_array

    def dump_struct(self, value, write, escape=escape):
        i = id(value)
        if i in self.memo:
            raise TypeError('cannot marshal recursive dictionaries')
        self.memo[i] = None
        dump = self.__dump
        write('<value><struct>\n')
        for k, v in value.items():
            write('<member>\n')
            if not isinstance(k, str):
                raise TypeError('dictionary key must be string')
            write('<name>%s</name>\n' % escape(k))
            dump(v, write)
            write('</member>\n')
        write('</struct></value>\n')
        del self.memo[i]
    dispatch[dict] = dump_struct

    def dump_datetime(self, value, write):
        write('<value><dateTime.iso8601>')
        write(_strftime(value))
        write('</dateTime.iso8601></value>\n')
    dispatch[datetime] = dump_datetime

    def dump_instance(self, value, write):
        if value.__class__ in WRAPPERS:
            self.write = write
            value.encode(self)
            del self.write
        else:
            self.dump_struct(value.__dict__, write)
    dispatch[DateTime] = dump_instance
    dispatch[Binary] = dump_instance
    dispatch['_arbitrary_instance'] = dump_instance