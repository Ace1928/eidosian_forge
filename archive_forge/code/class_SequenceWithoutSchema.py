import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
class SequenceWithoutSchema(BaseTestCase):

    def testGetItem(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s[0] = 'abc'
        assert s['field-0']
        assert s[0]
        try:
            s['field-1']
        except KeyError:
            pass
        else:
            assert False, 'KeyError not raised'

    def testSetItem(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s['field-0'] = 'xxx'
        try:
            s['field-1'] = 'xxx'
        except KeyError:
            pass
        else:
            assert False, 'KeyError not raised'

    def testIter(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s.setComponentByPosition(1, univ.Integer(123))
        assert list(s) == ['field-0', 'field-1']

    def testKeys(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s.setComponentByPosition(1, univ.Integer(123))
        assert list(s.keys()) == ['field-0', 'field-1']

    def testValues(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s.setComponentByPosition(1, univ.Integer(123))
        assert list(s.values()) == [str2octs('abc'), 123]

    def testItems(self):
        s = univ.Sequence()
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s.setComponentByPosition(1, univ.Integer(123))
        assert list(s.items()) == [('field-0', str2octs('abc')), ('field-1', 123)]

    def testUpdate(self):
        s = univ.Sequence()
        assert not s
        s.setComponentByPosition(0, univ.OctetString('abc'))
        s.setComponentByPosition(1, univ.Integer(123))
        assert s
        assert list(s.keys()) == ['field-0', 'field-1']
        assert list(s.values()) == [str2octs('abc'), 123]
        assert list(s.items()) == [('field-0', str2octs('abc')), ('field-1', 123)]
        s['field-0'] = univ.OctetString('def')
        assert list(s.values()) == [str2octs('def'), 123]
        s['field-1'] = univ.OctetString('ghi')
        assert list(s.values()) == [str2octs('def'), str2octs('ghi')]
        try:
            s['field-2'] = univ.OctetString('xxx')
        except KeyError:
            pass
        else:
            assert False, 'unknown field at schema-less object tolerated'
        assert 'field-0' in s
        s.clear()
        assert 'field-0' not in s