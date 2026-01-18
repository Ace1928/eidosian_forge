import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
class DulwichHttpClientTest(CompatTestCase, DulwichClientTestBase):
    min_git_version = (1, 7, 0, 2)

    def setUp(self):
        CompatTestCase.setUp(self)
        DulwichClientTestBase.setUp(self)
        self._httpd = HTTPGitServer(('localhost', 0), self.gitroot)
        self.addCleanup(self._httpd.shutdown)
        threading.Thread(target=self._httpd.serve_forever).start()
        run_git_or_fail(['config', 'http.uploadpack', 'true'], cwd=self.dest)
        run_git_or_fail(['config', 'http.receivepack', 'true'], cwd=self.dest)

    def tearDown(self):
        DulwichClientTestBase.tearDown(self)
        CompatTestCase.tearDown(self)
        self._httpd.shutdown()
        self._httpd.socket.close()

    def _client(self):
        return client.HttpGitClient(self._httpd.get_url())

    def _build_path(self, path):
        return path

    def test_archive(self):
        raise SkipTest('exporting archives not supported over http')