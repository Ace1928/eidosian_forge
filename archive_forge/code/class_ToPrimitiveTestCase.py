import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
class ToPrimitiveTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(ToPrimitiveTestCase, self).setUp()
        self.trans_fixture = self.useFixture(fixture.Translation())

    def test_bytes(self):
        self.assertEqual(jsonutils.to_primitive(b'abc'), 'abc')

    def test_list(self):
        self.assertEqual([1, 2, 3], jsonutils.to_primitive([1, 2, 3]))

    def test_empty_list(self):
        self.assertEqual([], jsonutils.to_primitive([]))

    def test_tuple(self):
        self.assertEqual([1, 2, 3], jsonutils.to_primitive((1, 2, 3)))

    def test_dict(self):
        self.assertEqual(dict(a=1, b=2, c=3), jsonutils.to_primitive(dict(a=1, b=2, c=3)))

    def test_empty_dict(self):
        self.assertEqual({}, jsonutils.to_primitive({}))

    def test_datetime(self):
        x = datetime.datetime(1920, 2, 3, 4, 5, 6, 7)
        self.assertEqual('1920-02-03T04:05:06.000007', jsonutils.to_primitive(x))

    def test_datetime_preserve(self):
        x = datetime.datetime(1920, 2, 3, 4, 5, 6, 7)
        self.assertEqual(x, jsonutils.to_primitive(x, convert_datetime=False))

    def test_DateTime(self):
        x = xmlrpclib.DateTime()
        x.decode('19710203T04:05:06')
        self.assertEqual('1971-02-03T04:05:06.000000', jsonutils.to_primitive(x))

    def test_iter(self):

        class IterClass(object):

            def __init__(self):
                self.data = [1, 2, 3, 4, 5]
                self.index = 0

            def __iter__(self):
                return self

            def next(self):
                if self.index == len(self.data):
                    raise StopIteration
                self.index = self.index + 1
                return self.data[self.index - 1]
            __next__ = next
        x = IterClass()
        self.assertEqual([1, 2, 3, 4, 5], jsonutils.to_primitive(x))

    def test_iteritems(self):

        class IterItemsClass(object):

            def __init__(self):
                self.data = dict(a=1, b=2, c=3).items()
                self.index = 0

            def iteritems(self):
                return self.data
        x = IterItemsClass()
        p = jsonutils.to_primitive(x)
        self.assertEqual({'a': 1, 'b': 2, 'c': 3}, p)

    def test_iteritems_with_cycle(self):

        class IterItemsClass(object):

            def __init__(self):
                self.data = dict(a=1, b=2, c=3)
                self.index = 0

            def iteritems(self):
                return self.data.items()
        x = IterItemsClass()
        x2 = IterItemsClass()
        x.data['other'] = x2
        x2.data['other'] = x
        jsonutils.to_primitive(x)

    def test_items(self):

        class ItemsClass(object):

            def __init__(self):
                self.data = dict(a=1, b=2, c=3)

            def items(self):
                return self.data.items()
        x = ItemsClass()
        p = jsonutils.to_primitive(x)
        self.assertEqual({'a': 1, 'b': 2, 'c': 3}, p)

    def test_precedence_items_iteritems(self):

        class ItemsIterItemsClass(object):

            def items(self):
                return {'items': 'items'}

            def iteritems(self):
                return {'iteritems': 'iteritems'}
        x = ItemsIterItemsClass()
        p = jsonutils.to_primitive(x)
        self.assertEqual({'iteritems': 'iteritems'}, p)

    def test_mapping(self):

        class MappingClass(collections.abc.Mapping):

            def __init__(self):
                self.data = dict(a=1, b=2, c=3)

            def __getitem__(self, val):
                return self.data[val]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)
        x = MappingClass()
        p = jsonutils.to_primitive(x)
        self.assertEqual({'a': 1, 'b': 2, 'c': 3}, p)

    def test_instance(self):

        class MysteryClass(object):
            a = 10

            def __init__(self):
                self.b = 1
        x = MysteryClass()
        self.assertEqual(dict(b=1), jsonutils.to_primitive(x, convert_instances=True))
        self.assertRaises(ValueError, jsonutils.to_primitive, x)

    def test_typeerror(self):
        x = bytearray
        self.assertEqual(u"<class 'bytearray'>", jsonutils.to_primitive(x))

    def test_nasties(self):

        def foo():
            pass
        x = [datetime, foo, dir]
        ret = jsonutils.to_primitive(x)
        self.assertEqual(3, len(ret))
        self.assertTrue(ret[0].startswith(u"<module 'datetime' from ") or ret[0].startswith(u"<module 'datetime' (built-in)"))
        self.assertTrue(ret[1].startswith('<function ToPrimitiveTestCase.test_nasties.<locals>.foo at 0x'))
        self.assertEqual('<built-in function dir>', ret[2])

    def test_depth(self):

        class LevelsGenerator(object):

            def __init__(self, levels):
                self._levels = levels

            def iteritems(self):
                if self._levels == 0:
                    return iter([])
                else:
                    return iter([(0, LevelsGenerator(self._levels - 1))])
        l4_obj = LevelsGenerator(4)
        json_l2 = {0: {0: None}}
        json_l3 = {0: {0: {0: None}}}
        json_l4 = {0: {0: {0: {0: None}}}}
        ret = jsonutils.to_primitive(l4_obj, max_depth=2)
        self.assertEqual(json_l2, ret)
        ret = jsonutils.to_primitive(l4_obj, max_depth=3)
        self.assertEqual(json_l3, ret)
        ret = jsonutils.to_primitive(l4_obj, max_depth=4)
        self.assertEqual(json_l4, ret)

    def test_ipaddr_using_netaddr(self):
        thing = {'ip_addr': netaddr.IPAddress('1.2.3.4')}
        ret = jsonutils.to_primitive(thing)
        self.assertEqual({'ip_addr': '1.2.3.4'}, ret)

    def test_ipaddr_using_ipaddress_v4(self):
        thing = {'ip_addr': ipaddress.ip_address('192.168.0.1')}
        ret = jsonutils.to_primitive(thing)
        self.assertEqual({'ip_addr': '192.168.0.1'}, ret)

    def test_ipaddr_using_ipaddress_v6(self):
        thing = {'ip_addr': ipaddress.ip_address('2001:db8::')}
        ret = jsonutils.to_primitive(thing)
        self.assertEqual({'ip_addr': '2001:db8::'}, ret)

    def test_ipnet_using_netaddr(self):
        thing = {'ip_net': netaddr.IPNetwork('1.2.3.0/24')}
        ret = jsonutils.to_primitive(thing)
        self.assertEqual({'ip_net': '1.2.3.0/24'}, ret)

    def test_message_with_param(self):
        msg = self.trans_fixture.lazy('A message with param: %s')
        msg = msg % 'test_domain'
        ret = jsonutils.to_primitive(msg)
        self.assertEqual(msg, ret)

    def test_message_with_named_param(self):
        msg = self.trans_fixture.lazy('A message with params: %(param)s')
        msg = msg % {'param': 'hello'}
        ret = jsonutils.to_primitive(msg)
        self.assertEqual(msg, ret)

    def test_fallback(self):
        obj = ReprObject()
        self.assertRaises(ValueError, jsonutils.to_primitive, obj)
        ret = jsonutils.to_primitive(obj, fallback=repr)
        self.assertEqual('repr', ret)

    def test_fallback_list(self):
        obj = ReprObject()
        obj_list = [obj]
        self.assertRaises(ValueError, jsonutils.to_primitive, obj_list)
        ret = jsonutils.to_primitive(obj_list, fallback=repr)
        self.assertEqual(['repr'], ret)

    def test_fallback_itertools_count(self):
        obj = itertools.count(1)
        ret = jsonutils.to_primitive(obj)
        self.assertEqual(str(obj), ret)
        ret = jsonutils.to_primitive(obj, fallback=lambda _: 'itertools_count')
        self.assertEqual('itertools_count', ret)

    def test_fallback_nasty(self):
        obj = int
        ret = jsonutils.to_primitive(obj)
        self.assertEqual(str(obj), ret)

        def formatter(typeobj):
            return 'type:%s' % typeobj.__name__
        ret = jsonutils.to_primitive(obj, fallback=formatter)
        self.assertEqual('type:int', ret)

    def test_fallback_typeerror(self):

        class NotIterable(object):
            __iter__ = None
        obj = NotIterable()
        ret = jsonutils.to_primitive(obj)
        self.assertEqual(str(obj), ret)
        ret = jsonutils.to_primitive(obj, fallback=lambda _: 'fallback')
        self.assertEqual('fallback', ret)

    def test_fallback_typeerror_IO_object(self):
        obj = io.IOBase
        ret = jsonutils.to_primitive(obj)
        self.assertEqual(str(obj), ret)
        ret = jsonutils.to_primitive(obj, fallback=lambda _: 'fallback')
        self.assertEqual('fallback', ret)

    def test_exception(self):
        self.assertIn(jsonutils.to_primitive(ValueError('an exception')), ["ValueError('an exception',)", "ValueError('an exception')"])