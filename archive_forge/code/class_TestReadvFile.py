from io import BytesIO
from ... import tests
from .. import pack
class TestReadvFile(tests.TestCaseWithTransport):
    """Tests of the ReadVFile class.

    Error cases are deliberately undefined: this code adapts the underlying
    transport interface to a single 'streaming read' interface as
    ContainerReader needs.
    """

    def test_read_bytes(self):
        """Test reading of both single bytes and all bytes in a hunk."""
        transport = self.get_transport()
        transport.put_bytes('sample', b'0123456789')
        f = pack.ReadVFile(transport.readv('sample', [(0, 1), (1, 2), (4, 1), (6, 2)]))
        results = []
        results.append(f.read(1))
        results.append(f.read(2))
        results.append(f.read(1))
        results.append(f.read(1))
        results.append(f.read(1))
        self.assertEqual([b'0', b'12', b'4', b'6', b'7'], results)

    def test_readline(self):
        """Test using readline() as ContainerReader does.

        This is always within a readv hunk, never across it.
        """
        transport = self.get_transport()
        transport.put_bytes('sample', b'0\n2\n4\n')
        f = pack.ReadVFile(transport.readv('sample', [(0, 2), (2, 4)]))
        results = []
        results.append(f.readline())
        results.append(f.readline())
        results.append(f.readline())
        self.assertEqual([b'0\n', b'2\n', b'4\n'], results)

    def test_readline_and_read(self):
        """Test exercising one byte reads, readline, and then read again."""
        transport = self.get_transport()
        transport.put_bytes('sample', b'0\n2\n4\n')
        f = pack.ReadVFile(transport.readv('sample', [(0, 6)]))
        results = []
        results.append(f.read(1))
        results.append(f.readline())
        results.append(f.read(4))
        self.assertEqual([b'0', b'\n', b'2\n4\n'], results)