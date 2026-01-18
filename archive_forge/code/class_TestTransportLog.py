from breezy import transport
from breezy.tests import TestCaseWithMemoryTransport
from breezy.trace import mutter
from breezy.transport.log import TransportLogDecorator
class TestTransportLog(TestCaseWithMemoryTransport):

    def test_log_transport(self):
        base_transport = self.get_transport('')
        logging_transport = transport.get_transport('log+' + base_transport.base)
        mutter('where are you?')
        logging_transport.mkdir('subdir')
        log = self.get_log()
        self.assertContainsRe(log, 'mkdir subdir')
        self.assertContainsRe(log, '  --> None')
        self.assertTrue(logging_transport.has('subdir'))
        self.assertTrue(base_transport.has('subdir'))

    def test_log_readv(self):
        base_transport = DummyReadvTransport()
        logging_transport = TransportLogDecorator('log+dummy:///', _decorated=base_transport)
        result = base_transport.readv('foo', [(0, 10)])
        next(result)
        result = logging_transport.readv('foo', [(0, 10)])
        self.assertEqual(list(result), [(0, 'abcdefghij')])