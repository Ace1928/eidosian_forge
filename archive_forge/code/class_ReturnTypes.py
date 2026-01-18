import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class ReturnTypes(object):

    @expose(wsme.types.bytes)
    def getbytes(self):
        return b'astring'

    @expose(wsme.types.text)
    def gettext(self):
        return 'ã\x81®'

    @expose(int)
    def getint(self):
        return 2

    @expose(float)
    def getfloat(self):
        return 3.14159265

    @expose(decimal.Decimal)
    def getdecimal(self):
        return decimal.Decimal('3.14159265')

    @expose(datetime.date)
    def getdate(self):
        return datetime.date(1994, 1, 26)

    @expose(bool)
    def getbooltrue(self):
        return True

    @expose(bool)
    def getboolfalse(self):
        return False

    @expose(datetime.time)
    def gettime(self):
        return datetime.time(12, 0, 0)

    @expose(datetime.datetime)
    def getdatetime(self):
        return datetime.datetime(1994, 1, 26, 12, 0, 0)

    @expose(wsme.types.binary)
    def getbinary(self):
        return binarysample

    @expose(NestedOuter)
    def getnested(self):
        n = NestedOuter()
        return n

    @expose([wsme.types.bytes])
    def getbytesarray(self):
        return [b'A', b'B', b'C']

    @expose([NestedOuter])
    def getnestedarray(self):
        return [NestedOuter(), NestedOuter()]

    @expose({wsme.types.bytes: NestedOuter})
    def getnesteddict(self):
        return {b'a': NestedOuter(), b'b': NestedOuter()}

    @expose(NestedOuter)
    def getobjectarrayattribute(self):
        obj = NestedOuter()
        obj.inner_array = [NestedInner(12), NestedInner(13)]
        return obj

    @expose(NestedOuter)
    def getobjectdictattribute(self):
        obj = NestedOuter()
        obj.inner_dict = {'12': NestedInner(12), '13': NestedInner(13)}
        return obj

    @expose(myenumtype)
    def getenum(self):
        return b'v2'

    @expose(NamedAttrsObject)
    def getnamedattrsobj(self):
        return NamedAttrsObject(5, 6)