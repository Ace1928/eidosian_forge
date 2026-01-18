from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class BufferedPktLineWriterTests(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self._output = BytesIO()
        self._writer = BufferedPktLineWriter(self._output.write, bufsize=16)

    def assertOutputEquals(self, expected):
        self.assertEqual(expected, self._output.getvalue())

    def _truncate(self):
        self._output.seek(0)
        self._output.truncate()

    def test_write(self):
        self._writer.write(b'foo')
        self.assertOutputEquals(b'')
        self._writer.flush()
        self.assertOutputEquals(b'0007foo')

    def test_write_none(self):
        self._writer.write(None)
        self.assertOutputEquals(b'')
        self._writer.flush()
        self.assertOutputEquals(b'0000')

    def test_flush_empty(self):
        self._writer.flush()
        self.assertOutputEquals(b'')

    def test_write_multiple(self):
        self._writer.write(b'foo')
        self._writer.write(b'bar')
        self.assertOutputEquals(b'')
        self._writer.flush()
        self.assertOutputEquals(b'0007foo0007bar')

    def test_write_across_boundary(self):
        self._writer.write(b'foo')
        self._writer.write(b'barbaz')
        self.assertOutputEquals(b'0007foo000abarba')
        self._truncate()
        self._writer.flush()
        self.assertOutputEquals(b'z')

    def test_write_to_boundary(self):
        self._writer.write(b'foo')
        self._writer.write(b'barba')
        self.assertOutputEquals(b'0007foo0009barba')
        self._truncate()
        self._writer.write(b'z')
        self._writer.flush()
        self.assertOutputEquals(b'0005z')