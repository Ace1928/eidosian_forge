import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
class TestCmdServeChrooting(TestBzrServeBase):

    def test_serve_tcp(self):
        """'brz serve' wraps the given --directory in a ChrootServer.

        So requests that search up through the parent directories (like
        find_repositoryV3) will give "not found" responses, rather than
        InvalidURLJoin or jail break errors.
        """
        t = self.get_transport()
        t.mkdir('server-root')
        self.run_bzr_serve_then_func(['--listen', '127.0.0.1', '--port', '0', '--directory', t.local_abspath('server-root'), '--allow-writes'], func=self.when_server_started)
        self.assertEqual((b'norepository',), self.client_resp)

    def when_server_started(self):
        client_medium = medium.SmartTCPClientMedium('127.0.0.1', self.tcp_server.port, 'bzr://localhost:%d/' % (self.tcp_server.port,))
        smart_client = client._SmartClient(client_medium)
        resp = smart_client.call('mkdir', 'foo', '')
        resp = smart_client.call('BzrDirFormat.initialize', 'foo/')
        try:
            resp = smart_client.call('BzrDir.find_repositoryV3', 'foo/')
        except errors.ErrorFromSmartServer as e:
            resp = e.error_tuple
        self.client_resp = resp
        client_medium.disconnect()