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
class DulwichClientTestBase:
    """Tests for client/server compatibility."""

    def setUp(self):
        self.gitroot = os.path.dirname(import_repo_to_dir('server_new.export').rstrip(os.sep))
        self.dest = os.path.join(self.gitroot, 'dest')
        file.ensure_dir_exists(self.dest)
        run_git_or_fail(['init', '--quiet', '--bare'], cwd=self.dest)

    def tearDown(self):
        rmtree_ro(self.gitroot)

    def assertDestEqualsSrc(self):
        repo_dir = os.path.join(self.gitroot, 'server_new.export')
        dest_repo_dir = os.path.join(self.gitroot, 'dest')
        with repo.Repo(repo_dir) as src:
            with repo.Repo(dest_repo_dir) as dest:
                self.assertReposEqual(src, dest)

    def _client(self):
        raise NotImplementedError

    def _build_path(self):
        raise NotImplementedError

    def _do_send_pack(self):
        c = self._client()
        srcpath = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(srcpath) as src:
            sendrefs = dict(src.get_refs())
            del sendrefs[b'HEAD']
            c.send_pack(self._build_path('/dest'), lambda _: sendrefs, src.generate_pack_data)

    def test_send_pack(self):
        self._do_send_pack()
        self.assertDestEqualsSrc()

    def test_send_pack_nothing_to_send(self):
        self._do_send_pack()
        self.assertDestEqualsSrc()
        self._do_send_pack()

    @staticmethod
    def _add_file(repo, tree_id, filename, contents):
        tree = repo[tree_id]
        blob = objects.Blob()
        blob.data = contents.encode('utf-8')
        repo.object_store.add_object(blob)
        tree.add(filename.encode('utf-8'), stat.S_IFREG | 420, blob.id)
        repo.object_store.add_object(tree)
        return tree.id

    def test_send_pack_from_shallow_clone(self):
        c = self._client()
        server_new_path = os.path.join(self.gitroot, 'server_new.export')
        run_git_or_fail(['config', 'http.uploadpack', 'true'], cwd=server_new_path)
        run_git_or_fail(['config', 'http.receivepack', 'true'], cwd=server_new_path)
        remote_path = self._build_path('/server_new.export')
        with repo.Repo(self.dest) as local:
            result = c.fetch(remote_path, local, depth=1)
            for r in result.refs.items():
                local.refs.set_if_equals(r[0], None, r[1])
            tree_id = local[local.head()].tree
            for filename, contents in [('bar', 'bar contents'), ('zop', 'zop contents')]:
                tree_id = self._add_file(local, tree_id, filename, contents)
                commit_id = local.do_commit(message=b'add ' + filename.encode('utf-8'), committer=b'Joe Example <joe@example.com>', tree=tree_id)
            sendrefs = dict(local.get_refs())
            del sendrefs[b'HEAD']
            c.send_pack(remote_path, lambda _: sendrefs, local.generate_pack_data)
        with repo.Repo(server_new_path) as remote:
            self.assertEqual(remote.head(), commit_id)

    def test_send_without_report_status(self):
        c = self._client()
        c._send_capabilities.remove(b'report-status')
        srcpath = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(srcpath) as src:
            sendrefs = dict(src.get_refs())
            del sendrefs[b'HEAD']
            c.send_pack(self._build_path('/dest'), lambda _: sendrefs, src.generate_pack_data)
            self.assertDestEqualsSrc()

    def make_dummy_commit(self, dest):
        b = objects.Blob.from_string(b'hi')
        dest.object_store.add_object(b)
        t = index.commit_tree(dest.object_store, [(b'hi', b.id, 33188)])
        c = objects.Commit()
        c.author = c.committer = b'Foo Bar <foo@example.com>'
        c.author_time = c.commit_time = 0
        c.author_timezone = c.commit_timezone = 0
        c.message = b'hi'
        c.tree = t
        dest.object_store.add_object(c)
        return c.id

    def disable_ff_and_make_dummy_commit(self):
        dest = repo.Repo(os.path.join(self.gitroot, 'dest'))
        run_git_or_fail(['config', 'receive.denyNonFastForwards', 'true'], cwd=dest.path)
        commit_id = self.make_dummy_commit(dest)
        return (dest, commit_id)

    def compute_send(self, src):
        sendrefs = dict(src.get_refs())
        del sendrefs[b'HEAD']
        return (sendrefs, src.generate_pack_data)

    def test_send_pack_one_error(self):
        dest, dummy_commit = self.disable_ff_and_make_dummy_commit()
        dest.refs[b'refs/heads/master'] = dummy_commit
        repo_dir = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(repo_dir) as src:
            sendrefs, gen_pack = self.compute_send(src)
            c = self._client()
            result = c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
            self.assertEqual({b'refs/heads/branch': None, b'refs/heads/master': 'non-fast-forward'}, result.ref_status)

    def test_send_pack_multiple_errors(self):
        dest, dummy = self.disable_ff_and_make_dummy_commit()
        branch, master = (b'refs/heads/branch', b'refs/heads/master')
        dest.refs[branch] = dest.refs[master] = dummy
        repo_dir = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(repo_dir) as src:
            sendrefs, gen_pack = self.compute_send(src)
            c = self._client()
            result = c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
            self.assertEqual({branch: 'non-fast-forward', master: 'non-fast-forward'}, result.ref_status)

    def test_archive(self):
        c = self._client()
        f = BytesIO()
        c.archive(self._build_path('/server_new.export'), b'HEAD', f.write)
        f.seek(0)
        tf = tarfile.open(fileobj=f)
        self.assertEqual(['baz', 'foo'], tf.getnames())

    def test_fetch_pack(self):
        c = self._client()
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

    def test_fetch_pack_depth(self):
        c = self._client()
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest, depth=1)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertEqual(dest.get_shallow(), {b'35e0b59e187dd72a0af294aedffc213eaa4d03ff', b'514dc6d3fbfe77361bcaef320c4d21b72bc10be9'})

    def test_repeat(self):
        c = self._client()
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()
            result = c.fetch(self._build_path('/server_new.export'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

    def test_fetch_empty_pack(self):
        c = self._client()
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

            def dw(refs, **kwargs):
                return list(refs.values())
            result = c.fetch(self._build_path('/server_new.export'), dest, determine_wants=dw)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

    def test_incremental_fetch_pack(self):
        self.test_fetch_pack()
        dest, dummy = self.disable_ff_and_make_dummy_commit()
        dest.refs[b'refs/heads/master'] = dummy
        c = self._client()
        repo_dir = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(repo_dir) as dest:
            result = c.fetch(self._build_path('/dest'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

    def test_fetch_pack_no_side_band_64k(self):
        c = self._client()
        c._fetch_capabilities.remove(b'side-band-64k')
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest)
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])
            self.assertDestEqualsSrc()

    def test_fetch_pack_zero_sha(self):
        c = self._client()
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            result = c.fetch(self._build_path('/server_new.export'), dest, lambda refs, **kwargs: [protocol.ZERO_SHA])
            for r in result.refs.items():
                dest.refs.set_if_equals(r[0], None, r[1])

    def test_send_remove_branch(self):
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            dummy_commit = self.make_dummy_commit(dest)
            dest.refs[b'refs/heads/master'] = dummy_commit
            dest.refs[b'refs/heads/abranch'] = dummy_commit
            sendrefs = dict(dest.refs)
            sendrefs[b'refs/heads/abranch'] = b'00' * 20
            del sendrefs[b'HEAD']

            def gen_pack(have, want, ofs_delta=False, progress=None):
                return (0, [])
            c = self._client()
            self.assertEqual(dest.refs[b'refs/heads/abranch'], dummy_commit)
            c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
            self.assertNotIn(b'refs/heads/abranch', dest.refs)

    def test_send_new_branch_empty_pack(self):
        with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
            dummy_commit = self.make_dummy_commit(dest)
            dest.refs[b'refs/heads/master'] = dummy_commit
            dest.refs[b'refs/heads/abranch'] = dummy_commit
            sendrefs = {b'refs/heads/bbranch': dummy_commit}

            def gen_pack(have, want, ofs_delta=False, progress=None):
                return (0, [])
            c = self._client()
            self.assertEqual(dest.refs[b'refs/heads/abranch'], dummy_commit)
            c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
            self.assertEqual(dummy_commit, dest.refs[b'refs/heads/abranch'])

    def test_get_refs(self):
        c = self._client()
        refs = c.get_refs(self._build_path('/server_new.export'))
        repo_dir = os.path.join(self.gitroot, 'server_new.export')
        with repo.Repo(repo_dir) as dest:
            self.assertDictEqual(dest.refs.as_dict(), refs)