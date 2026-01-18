import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def _checkout_remote_branch(self):
    errstream = BytesIO()
    outstream = BytesIO()
    porcelain.commit(repo=self.repo.path, message=b'init', author=b'author <email>', committer=b'committer <email>')
    clone_path = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, clone_path)
    target_repo = porcelain.clone(self.repo.path, target=clone_path, errstream=errstream)
    try:
        self.assertEqual(target_repo[b'HEAD'], self.repo[b'HEAD'])
    finally:
        target_repo.close()
    handle, fullpath = tempfile.mkstemp(dir=clone_path)
    os.close(handle)
    porcelain.add(repo=clone_path, paths=[fullpath])
    porcelain.commit(repo=clone_path, message=b'push', author=b'author <email>', committer=b'committer <email>')
    refs_path = b'refs/heads/foo'
    new_id = self.repo[b'HEAD'].id
    self.assertNotEqual(new_id, ZERO_SHA)
    self.repo.refs[refs_path] = new_id
    porcelain.push(clone_path, 'origin', b'HEAD:' + refs_path, outstream=outstream, errstream=errstream)
    self.assertEqual(target_repo.refs[b'refs/remotes/origin/foo'], target_repo.refs[b'HEAD'])
    porcelain.checkout_branch(target_repo, b'origin/foo')
    original_id = target_repo[b'HEAD'].id
    uni_id = target_repo[b'refs/remotes/origin/uni'].id
    expected_refs = {b'HEAD': original_id, b'refs/heads/master': original_id, b'refs/heads/foo': original_id, b'refs/remotes/origin/foo': original_id, b'refs/remotes/origin/uni': uni_id, b'refs/remotes/origin/HEAD': new_id, b'refs/remotes/origin/master': new_id}
    self.assertEqual(expected_refs, target_repo.get_refs())
    return target_repo