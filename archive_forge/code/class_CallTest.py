import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class CallTest(unittest.TestCase):

    def test_call_with_call(self):
        kall = _Call()
        self.assertEqual(kall, _Call())
        self.assertEqual(kall, _Call(('',)))
        self.assertEqual(kall, _Call(((),)))
        self.assertEqual(kall, _Call(({},)))
        self.assertEqual(kall, _Call(('', ())))
        self.assertEqual(kall, _Call(('', {})))
        self.assertEqual(kall, _Call(('', (), {})))
        self.assertEqual(kall, _Call(('foo',)))
        self.assertEqual(kall, _Call(('bar', ())))
        self.assertEqual(kall, _Call(('baz', {})))
        self.assertEqual(kall, _Call(('spam', (), {})))
        kall = _Call(((1, 2, 3),))
        self.assertEqual(kall, _Call(((1, 2, 3),)))
        self.assertEqual(kall, _Call(('', (1, 2, 3))))
        self.assertEqual(kall, _Call(((1, 2, 3), {})))
        self.assertEqual(kall, _Call(('', (1, 2, 3), {})))
        kall = _Call(((1, 2, 4),))
        self.assertNotEqual(kall, _Call(('', (1, 2, 3))))
        self.assertNotEqual(kall, _Call(('', (1, 2, 3), {})))
        kall = _Call(('foo', (1, 2, 4)))
        self.assertNotEqual(kall, _Call(('', (1, 2, 4))))
        self.assertNotEqual(kall, _Call(('', (1, 2, 4), {})))
        self.assertNotEqual(kall, _Call(('bar', (1, 2, 4))))
        self.assertNotEqual(kall, _Call(('bar', (1, 2, 4), {})))
        kall = _Call(({'a': 3},))
        self.assertEqual(kall, _Call(('', (), {'a': 3})))
        self.assertEqual(kall, _Call(('', {'a': 3})))
        self.assertEqual(kall, _Call(((), {'a': 3})))
        self.assertEqual(kall, _Call(({'a': 3},)))

    def test_empty__Call(self):
        args = _Call()
        self.assertEqual(args, ())
        self.assertEqual(args, ('foo',))
        self.assertEqual(args, ((),))
        self.assertEqual(args, ('foo', ()))
        self.assertEqual(args, ('foo', (), {}))
        self.assertEqual(args, ('foo', {}))
        self.assertEqual(args, ({},))

    def test_named_empty_call(self):
        args = _Call(('foo', (), {}))
        self.assertEqual(args, ('foo',))
        self.assertEqual(args, ('foo', ()))
        self.assertEqual(args, ('foo', (), {}))
        self.assertEqual(args, ('foo', {}))
        self.assertNotEqual(args, ((),))
        self.assertNotEqual(args, ())
        self.assertNotEqual(args, ({},))
        self.assertNotEqual(args, ('bar',))
        self.assertNotEqual(args, ('bar', ()))
        self.assertNotEqual(args, ('bar', {}))

    def test_call_with_args(self):
        args = _Call(((1, 2, 3), {}))
        self.assertEqual(args, ((1, 2, 3),))
        self.assertEqual(args, ('foo', (1, 2, 3)))
        self.assertEqual(args, ('foo', (1, 2, 3), {}))
        self.assertEqual(args, ((1, 2, 3), {}))

    def test_named_call_with_args(self):
        args = _Call(('foo', (1, 2, 3), {}))
        self.assertEqual(args, ('foo', (1, 2, 3)))
        self.assertEqual(args, ('foo', (1, 2, 3), {}))
        self.assertNotEqual(args, ((1, 2, 3),))
        self.assertNotEqual(args, ((1, 2, 3), {}))

    def test_call_with_kwargs(self):
        args = _Call(((), dict(a=3, b=4)))
        self.assertEqual(args, (dict(a=3, b=4),))
        self.assertEqual(args, ('foo', dict(a=3, b=4)))
        self.assertEqual(args, ('foo', (), dict(a=3, b=4)))
        self.assertEqual(args, ((), dict(a=3, b=4)))

    def test_named_call_with_kwargs(self):
        args = _Call(('foo', (), dict(a=3, b=4)))
        self.assertEqual(args, ('foo', dict(a=3, b=4)))
        self.assertEqual(args, ('foo', (), dict(a=3, b=4)))
        self.assertNotEqual(args, (dict(a=3, b=4),))
        self.assertNotEqual(args, ((), dict(a=3, b=4)))

    def test_call_with_args_call_empty_name(self):
        args = _Call(((1, 2, 3), {}))
        self.assertEqual(args, call(1, 2, 3))
        self.assertEqual(call(1, 2, 3), args)
        self.assertIn(call(1, 2, 3), [args])

    def test_call_ne(self):
        self.assertNotEqual(_Call(((1, 2, 3),)), call(1, 2))
        self.assertFalse(_Call(((1, 2, 3),)) != call(1, 2, 3))
        self.assertTrue(_Call(((1, 2), {})) != call(1, 2, 3))

    def test_call_non_tuples(self):
        kall = _Call(((1, 2, 3),))
        for value in (1, None, self, int):
            self.assertNotEqual(kall, value)
            self.assertFalse(kall == value)

    def test_repr(self):
        self.assertEqual(repr(_Call()), 'call()')
        self.assertEqual(repr(_Call(('foo',))), 'call.foo()')
        self.assertEqual(repr(_Call(((1, 2, 3), {'a': 'b'}))), "call(1, 2, 3, a='b')")
        self.assertEqual(repr(_Call(('bar', (1, 2, 3), {'a': 'b'}))), "call.bar(1, 2, 3, a='b')")
        self.assertEqual(repr(call), 'call')
        self.assertEqual(str(call), 'call')
        self.assertEqual(repr(call()), 'call()')
        self.assertEqual(repr(call(1)), 'call(1)')
        self.assertEqual(repr(call(zz='thing')), "call(zz='thing')")
        self.assertEqual(repr(call().foo), 'call().foo')
        self.assertEqual(repr(call(1).foo.bar(a=3).bing), 'call().foo.bar().bing')
        self.assertEqual(repr(call().foo(1, 2, a=3)), 'call().foo(1, 2, a=3)')
        self.assertEqual(repr(call()()), 'call()()')
        self.assertEqual(repr(call(1)(2)), 'call()(2)')
        self.assertEqual(repr(call()().bar().baz.beep(1)), 'call()().bar().baz.beep(1)')

    def test_call(self):
        self.assertEqual(call(), ('', (), {}))
        self.assertEqual(call('foo', 'bar', one=3, two=4), ('', ('foo', 'bar'), {'one': 3, 'two': 4}))
        mock = Mock()
        mock(1, 2, 3)
        mock(a=3, b=6)
        self.assertEqual(mock.call_args_list, [call(1, 2, 3), call(a=3, b=6)])

    def test_attribute_call(self):
        self.assertEqual(call.foo(1), ('foo', (1,), {}))
        self.assertEqual(call.bar.baz(fish='eggs'), ('bar.baz', (), {'fish': 'eggs'}))
        mock = Mock()
        mock.foo(1, 2, 3)
        mock.bar.baz(a=3, b=6)
        self.assertEqual(mock.method_calls, [call.foo(1, 2, 3), call.bar.baz(a=3, b=6)])

    def test_extended_call(self):
        result = call(1).foo(2).bar(3, a=4)
        self.assertEqual(result, ('().foo().bar', (3,), dict(a=4)))
        mock = MagicMock()
        mock(1, 2, a=3, b=4)
        self.assertEqual(mock.call_args, call(1, 2, a=3, b=4))
        self.assertNotEqual(mock.call_args, call(1, 2, 3))
        self.assertEqual(mock.call_args_list, [call(1, 2, a=3, b=4)])
        self.assertEqual(mock.mock_calls, [call(1, 2, a=3, b=4)])
        mock = MagicMock()
        mock.foo(1).bar()().baz.beep(a=6)
        last_call = call.foo(1).bar()().baz.beep(a=6)
        self.assertEqual(mock.mock_calls[-1], last_call)
        self.assertEqual(mock.mock_calls, last_call.call_list())

    def test_call_list(self):
        mock = MagicMock()
        mock(1)
        self.assertEqual(call(1).call_list(), mock.mock_calls)
        mock = MagicMock()
        mock(1).method(2)
        self.assertEqual(call(1).method(2).call_list(), mock.mock_calls)
        mock = MagicMock()
        mock(1).method(2)(3)
        self.assertEqual(call(1).method(2)(3).call_list(), mock.mock_calls)
        mock = MagicMock()
        int(mock(1).method(2)(3).foo.bar.baz(4)(5))
        kall = call(1).method(2)(3).foo.bar.baz(4)(5).__int__()
        self.assertEqual(kall.call_list(), mock.mock_calls)

    def test_call_any(self):
        self.assertEqual(call, ANY)
        m = MagicMock()
        int(m)
        self.assertEqual(m.mock_calls, [ANY])
        self.assertEqual([ANY], m.mock_calls)

    def test_two_args_call(self):
        args = _Call(((1, 2), {'a': 3}), two=True)
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], (1, 2))
        self.assertEqual(args[1], {'a': 3})
        other_args = _Call(((1, 2), {'a': 3}))
        self.assertEqual(args, other_args)