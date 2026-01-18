from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
class TestMockOpen(unittest.TestCase):

    def test_mock_open(self):
        mock = mock_open()
        with patch('%s.open' % __name__, mock, create=True) as patched:
            self.assertIs(patched, mock)
            open('foo')
        mock.assert_called_once_with('foo')

    def test_mock_open_context_manager(self):
        mock = mock_open()
        handle = mock.return_value
        with patch('%s.open' % __name__, mock, create=True):
            with open('foo') as f:
                f.read()
        expected_calls = [call('foo'), call().__enter__(), call().read(), call().__exit__(None, None, None)]
        self.assertEqual(mock.mock_calls, expected_calls)
        self.assertIs(f, handle)

    def test_mock_open_context_manager_multiple_times(self):
        mock = mock_open()
        with patch('%s.open' % __name__, mock, create=True):
            with open('foo') as f:
                f.read()
            with open('bar') as f:
                f.read()
        expected_calls = [call('foo'), call().__enter__(), call().read(), call().__exit__(None, None, None), call('bar'), call().__enter__(), call().read(), call().__exit__(None, None, None)]
        self.assertEqual(mock.mock_calls, expected_calls)

    def test_explicit_mock(self):
        mock = MagicMock()
        mock_open(mock)
        with patch('%s.open' % __name__, mock, create=True) as patched:
            self.assertIs(patched, mock)
            open('foo')
        mock.assert_called_once_with('foo')

    def test_read_data(self):
        mock = mock_open(read_data='foo')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            result = h.read()
        self.assertEqual(result, 'foo')

    def test_readline_data(self):
        mock = mock_open(read_data='foo\nbar\nbaz\n')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            line1 = h.readline()
            line2 = h.readline()
            line3 = h.readline()
        self.assertEqual(line1, 'foo\n')
        self.assertEqual(line2, 'bar\n')
        self.assertEqual(line3, 'baz\n')
        mock = mock_open(read_data='foo')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            result = h.readline()
        self.assertEqual(result, 'foo')

    def test_readlines_data(self):
        mock = mock_open(read_data='foo\nbar\nbaz\n')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            result = h.readlines()
        self.assertEqual(result, ['foo\n', 'bar\n', 'baz\n'])
        mock = mock_open(read_data='foo\nbar\nbaz')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            result = h.readlines()
        self.assertEqual(result, ['foo\n', 'bar\n', 'baz'])

    def test_read_bytes(self):
        mock = mock_open(read_data=b'\xc6')
        with patch('%s.open' % __name__, mock, create=True):
            with open('abc', 'rb') as f:
                result = f.read()
        self.assertEqual(result, b'\xc6')

    def test_readline_bytes(self):
        m = mock_open(read_data=b'abc\ndef\nghi\n')
        with patch('%s.open' % __name__, m, create=True):
            with open('abc', 'rb') as f:
                line1 = f.readline()
                line2 = f.readline()
                line3 = f.readline()
        self.assertEqual(line1, b'abc\n')
        self.assertEqual(line2, b'def\n')
        self.assertEqual(line3, b'ghi\n')

    def test_readlines_bytes(self):
        m = mock_open(read_data=b'abc\ndef\nghi\n')
        with patch('%s.open' % __name__, m, create=True):
            with open('abc', 'rb') as f:
                result = f.readlines()
        self.assertEqual(result, [b'abc\n', b'def\n', b'ghi\n'])

    def test_mock_open_read_with_argument(self):
        some_data = 'foo\nbar\nbaz'
        mock = mock_open(read_data=some_data)
        self.assertEqual(mock().read(10), some_data)

    def test_interleaved_reads(self):
        mock = mock_open(read_data='foo\nbar\nbaz\n')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            line1 = h.readline()
            rest = h.readlines()
        self.assertEqual(line1, 'foo\n')
        self.assertEqual(rest, ['bar\n', 'baz\n'])
        mock = mock_open(read_data='foo\nbar\nbaz\n')
        with patch('%s.open' % __name__, mock, create=True):
            h = open('bar')
            line1 = h.readline()
            rest = h.read()
        self.assertEqual(line1, 'foo\n')
        self.assertEqual(rest, 'bar\nbaz\n')

    def test_overriding_return_values(self):
        mock = mock_open(read_data='foo')
        handle = mock()
        handle.read.return_value = 'bar'
        handle.readline.return_value = 'bar'
        handle.readlines.return_value = ['bar']
        self.assertEqual(handle.read(), 'bar')
        self.assertEqual(handle.readline(), 'bar')
        self.assertEqual(handle.readlines(), ['bar'])
        self.assertEqual(handle.readline(), 'bar')
        self.assertEqual(handle.readline(), 'bar')