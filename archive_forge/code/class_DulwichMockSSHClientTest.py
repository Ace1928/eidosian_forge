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
class DulwichMockSSHClientTest(CompatTestCase, DulwichClientTestBase):

    def setUp(self):
        CompatTestCase.setUp(self)
        DulwichClientTestBase.setUp(self)
        self.real_vendor = client.get_ssh_vendor
        client.get_ssh_vendor = TestSSHVendor

    def tearDown(self):
        DulwichClientTestBase.tearDown(self)
        CompatTestCase.tearDown(self)
        client.get_ssh_vendor = self.real_vendor

    def _client(self):
        return client.SSHGitClient('localhost')

    def _build_path(self, path):
        return self.gitroot + path