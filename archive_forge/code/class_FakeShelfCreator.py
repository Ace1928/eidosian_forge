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
class FakeShelfCreator:

    def __init__(self, branch):
        self.branch = branch

    def write_shelf(self, shelf_file, message=None):
        tree = self.branch.repository.revision_tree(revision.NULL_REVISION)
        with tree.preview_transform() as tt:
            shelf.ShelfCreator._write_shelf(shelf_file, tt, revision.NULL_REVISION)