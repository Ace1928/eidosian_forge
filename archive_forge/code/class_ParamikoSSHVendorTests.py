import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
@skipIf(not has_paramiko, 'paramiko is not installed')
class ParamikoSSHVendorTests(TestCase):

    def setUp(self):
        import paramiko.transport
        if hasattr(paramiko.transport, 'SERVER_DISABLED_BY_GENTOO'):
            paramiko.transport.SERVER_DISABLED_BY_GENTOO = False
        self.commands = []
        socket.setdefaulttimeout(10)
        self.addCleanup(socket.setdefaulttimeout, None)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.listen(5)
        self.addCleanup(self.socket.close)
        self.port = self.socket.getsockname()[1]
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def tearDown(self):
        self.thread.join()

    def _run(self):
        try:
            conn, addr = self.socket.accept()
        except OSError:
            return False
        self.transport = paramiko.Transport(conn)
        self.addCleanup(self.transport.close)
        host_key = paramiko.RSAKey.from_private_key(StringIO(SERVER_KEY))
        self.transport.add_server_key(host_key)
        server = Server(self.commands)
        self.transport.start_server(server=server)

    def test_run_command_password(self):
        vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
        vendor.run_command('127.0.0.1', 'test_run_command_password', username=USER, port=self.port, password=PASSWORD)
        self.assertIn(b'test_run_command_password', self.commands)

    def test_run_command_with_privkey(self):
        key = paramiko.RSAKey.from_private_key(StringIO(CLIENT_KEY))
        vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
        vendor.run_command('127.0.0.1', 'test_run_command_with_privkey', username=USER, port=self.port, pkey=key)
        self.assertIn(b'test_run_command_with_privkey', self.commands)

    def test_run_command_data_transfer(self):
        vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
        con = vendor.run_command('127.0.0.1', 'test_run_command_data_transfer', username=USER, port=self.port, password=PASSWORD)
        self.assertIn(b'test_run_command_data_transfer', self.commands)
        channel = self.transport.accept(5)
        channel.send(b'stdout\n')
        channel.send_stderr(b'stderr\n')
        channel.close()
        self.assertEqual(b'stdout\n', con.read(4096))