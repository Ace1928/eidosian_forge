import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
class TestDecorator:

    def __init__(self):
        self._calls = []

    def lock_read(self):
        self._calls.append('lr')
        return lock.LogicalLockResult(self.unlock)

    def lock_write(self):
        self._calls.append('lw')
        return lock.LogicalLockResult(self.unlock)

    def unlock(self):
        self._calls.append('ul')

    def do_with_read(self):
        with self.lock_read():
            return 1

    def except_with_read(self):
        with self.lock_read():
            raise RuntimeError