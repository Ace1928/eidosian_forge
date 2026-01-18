from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class TestJsonifyGenericEncoder(PecanTestCase):

    def test_json_callable(self):

        class JsonCallable(object):

            def __init__(self, arg):
                self.arg = arg

            def __json__(self):
                return {'arg': self.arg}
        result = encode(JsonCallable('foo'))
        assert loads(result) == {'arg': 'foo'}

    def test_datetime(self):
        today = date.today()
        now = datetime.now()
        result = encode(today)
        assert loads(result) == str(today)
        result = encode(now)
        assert loads(result) == str(now)

    def test_decimal(self):
        d = Decimal('1.1')
        result = encode(d)
        assert loads(result) == float(d)

    def test_multidict(self):
        md = MultiDict()
        md.add('arg', 'foo')
        md.add('arg', 'bar')
        result = encode(md)
        assert loads(result) == {'arg': ['foo', 'bar']}

    def test_fallback_to_builtin_encoder(self):

        class Foo(object):
            pass
        self.assertRaises(TypeError, encode, Foo())