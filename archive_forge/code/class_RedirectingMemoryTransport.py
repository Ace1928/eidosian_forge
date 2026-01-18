import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class RedirectingMemoryTransport(memory.MemoryTransport):

    def mkdir(self, relpath, mode=None):
        if self._cwd == '/source/':
            raise errors.RedirectRequested(self.abspath(relpath), self.abspath('../target'), is_permanent=True)
        elif self._cwd == '/infinite-loop/':
            raise errors.RedirectRequested(self.abspath(relpath), self.abspath('../infinite-loop'), is_permanent=True)
        else:
            return super().mkdir(relpath, mode)

    def get(self, relpath):
        if self.clone(relpath)._cwd == '/infinite-loop/':
            raise errors.RedirectRequested(self.abspath(relpath), self.abspath('../infinite-loop'), is_permanent=True)
        else:
            return super().get(relpath)

    def _redirected_to(self, source, target):
        return transport.get_transport(target)