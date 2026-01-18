import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
@skipIfNonUnix
class TestMultiprocess(unittest.TestCase):

    def run_subproc(self, code: str) -> Tuple[str, str]:
        try:
            result = subprocess.run([sys.executable, '-Werror::DeprecationWarning'], capture_output=True, input=code, encoding='utf8', check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Process returned {e.returncode} stdout={e.stdout} stderr={e.stderr}') from e
        return (result.stdout, result.stderr)

    def test_listen_single(self):
        code = textwrap.dedent("\n            import asyncio\n            from tornado.tcpserver import TCPServer\n\n            async def main():\n                server = TCPServer()\n                server.listen(0, address='127.0.0.1')\n\n            asyncio.run(main())\n            print('012', end='')\n        ")
        out, err = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_bind_start(self):
        code = textwrap.dedent('\n            import warnings\n\n            from tornado.ioloop import IOLoop\n            from tornado.process import task_id\n            from tornado.tcpserver import TCPServer\n\n            warnings.simplefilter("ignore", DeprecationWarning)\n\n            server = TCPServer()\n            server.bind(0, address=\'127.0.0.1\')\n            server.start(3)\n            IOLoop.current().run_sync(lambda: None)\n            print(task_id(), end=\'\')\n        ')
        out, err = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_add_sockets(self):
        code = textwrap.dedent("\n            import asyncio\n            from tornado.netutil import bind_sockets\n            from tornado.process import fork_processes, task_id\n            from tornado.ioloop import IOLoop\n            from tornado.tcpserver import TCPServer\n\n            sockets = bind_sockets(0, address='127.0.0.1')\n            fork_processes(3)\n            async def post_fork_main():\n                server = TCPServer()\n                server.add_sockets(sockets)\n            asyncio.run(post_fork_main())\n            print(task_id(), end='')\n        ")
        out, err = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_listen_multi_reuse_port(self):
        code = textwrap.dedent("\n            import asyncio\n            import socket\n            from tornado.netutil import bind_sockets\n            from tornado.process import task_id, fork_processes\n            from tornado.tcpserver import TCPServer\n\n            # Pick an unused port which we will be able to bind to multiple times.\n            (sock,) = bind_sockets(0, address='127.0.0.1',\n                family=socket.AF_INET, reuse_port=True)\n            port = sock.getsockname()[1]\n\n            fork_processes(3)\n\n            async def main():\n                server = TCPServer()\n                server.listen(port, address='127.0.0.1', reuse_port=True)\n            asyncio.run(main())\n            print(task_id(), end='')\n            ")
        out, err = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')