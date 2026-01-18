import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class TestCallList(unittest.TestCase):

    def test_args_list_contains_call_list(self):
        mock = Mock()
        self.assertIsInstance(mock.call_args_list, _CallList)
        mock(1, 2)
        mock(a=3)
        mock(3, 4)
        mock(b=6)
        for kall in (call(1, 2), call(a=3), call(3, 4), call(b=6)):
            self.assertIn(kall, mock.call_args_list)
        calls = [call(a=3), call(3, 4)]
        self.assertIn(calls, mock.call_args_list)
        calls = [call(1, 2), call(a=3)]
        self.assertIn(calls, mock.call_args_list)
        calls = [call(3, 4), call(b=6)]
        self.assertIn(calls, mock.call_args_list)
        calls = [call(3, 4)]
        self.assertIn(calls, mock.call_args_list)
        self.assertNotIn(call('fish'), mock.call_args_list)
        self.assertNotIn([call('fish')], mock.call_args_list)

    def test_call_list_str(self):
        mock = Mock()
        mock(1, 2)
        mock.foo(a=3)
        mock.foo.bar().baz('fish', cat='dog')
        expected = "[call(1, 2),\n call.foo(a=3),\n call.foo.bar(),\n call.foo.bar().baz('fish', cat='dog')]"
        self.assertEqual(str(mock.mock_calls), expected)

    @unittest.skipIf(six.PY3, 'Unicode is properly handled with Python 3')
    def test_call_list_unicode(self):
        mock = Mock()

        class NonAsciiRepr(object):

            def __repr__(self):
                return 'é'
        mock(**{unicode('a'): NonAsciiRepr()})
        self.assertEqual(str(mock.mock_calls), '[call(a=é)]')

    def test_propertymock(self):
        p = patch('%s.SomeClass.one' % __name__, new_callable=PropertyMock)
        mock = p.start()
        try:
            SomeClass.one
            mock.assert_called_once_with()
            s = SomeClass()
            s.one
            mock.assert_called_with()
            self.assertEqual(mock.mock_calls, [call(), call()])
            s.one = 3
            self.assertEqual(mock.mock_calls, [call(), call(), call(3)])
        finally:
            p.stop()

    def test_propertymock_returnvalue(self):
        m = MagicMock()
        p = PropertyMock()
        type(m).foo = p
        returned = m.foo
        p.assert_called_once_with()
        self.assertIsInstance(returned, MagicMock)
        self.assertNotIsInstance(returned, PropertyMock)