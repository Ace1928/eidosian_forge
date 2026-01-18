import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class BlobReadTests(TestCase):
    """Test decompression of blobs."""

    def get_sha_file(self, cls, base, sha):
        dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', base)
        return cls.from_path(hex_to_filename(dir, sha))

    def get_blob(self, sha):
        """Return the blob named sha from the test data dir."""
        return self.get_sha_file(Blob, 'blobs', sha)

    def get_tree(self, sha):
        return self.get_sha_file(Tree, 'trees', sha)

    def get_tag(self, sha):
        return self.get_sha_file(Tag, 'tags', sha)

    def commit(self, sha):
        return self.get_sha_file(Commit, 'commits', sha)

    def test_decompress_simple_blob(self):
        b = self.get_blob(a_sha)
        self.assertEqual(b.data, b'test 1\n')
        self.assertEqual(b.sha().hexdigest().encode('ascii'), a_sha)

    def test_hash(self):
        b = self.get_blob(a_sha)
        self.assertEqual(hash(b.id), hash(b))

    def test_parse_empty_blob_object(self):
        sha = b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
        b = self.get_blob(sha)
        self.assertEqual(b.data, b'')
        self.assertEqual(b.id, sha)
        self.assertEqual(b.sha().hexdigest().encode('ascii'), sha)

    def test_create_blob_from_string(self):
        string = b'test 2\n'
        b = Blob.from_string(string)
        self.assertEqual(b.data, string)
        self.assertEqual(b.sha().hexdigest().encode('ascii'), b_sha)

    def test_legacy_from_file(self):
        b1 = Blob.from_string(b'foo')
        b_raw = b1.as_legacy_object()
        b2 = b1.from_file(BytesIO(b_raw))
        self.assertEqual(b1, b2)

    def test_legacy_from_file_compression_level(self):
        b1 = Blob.from_string(b'foo')
        b_raw = b1.as_legacy_object(compression_level=6)
        b2 = b1.from_file(BytesIO(b_raw))
        self.assertEqual(b1, b2)

    def test_chunks(self):
        string = b'test 5\n'
        b = Blob.from_string(string)
        self.assertEqual([string], b.chunked)

    def test_splitlines(self):
        for case in [[], [b'foo\nbar\n'], [b'bl\na', b'blie'], [b'bl\na', b'blie', b'bloe\n'], [b'', b'bl\na', b'blie', b'bloe\n'], [b'', b'', b'', b'bla\n'], [b'', b'', b'', b'bla\n', b''], [b'bl', b'', b'a\naaa'], [b'a\naaa', b'a']]:
            b = Blob()
            b.chunked = case
            self.assertEqual(b.data.splitlines(True), b.splitlines())

    def test_set_chunks(self):
        b = Blob()
        b.chunked = [b'te', b'st', b' 5\n']
        self.assertEqual(b'test 5\n', b.data)
        b.chunked = [b'te', b'st', b' 6\n']
        self.assertEqual(b'test 6\n', b.as_raw_string())
        self.assertEqual(b'test 6\n', bytes(b))

    def test_parse_legacy_blob(self):
        string = b'test 3\n'
        b = self.get_blob(c_sha)
        self.assertEqual(b.data, string)
        self.assertEqual(b.sha().hexdigest().encode('ascii'), c_sha)

    def test_eq(self):
        blob1 = self.get_blob(a_sha)
        blob2 = self.get_blob(a_sha)
        self.assertEqual(blob1, blob2)

    def test_read_tree_from_file(self):
        t = self.get_tree(tree_sha)
        self.assertEqual(t.items()[0], (b'a', 33188, a_sha))
        self.assertEqual(t.items()[1], (b'b', 33188, b_sha))

    def test_read_tree_from_file_parse_count(self):
        old_deserialize = Tree._deserialize

        def reset_deserialize():
            Tree._deserialize = old_deserialize
        self.addCleanup(reset_deserialize)
        self.deserialize_count = 0

        def counting_deserialize(*args, **kwargs):
            self.deserialize_count += 1
            return old_deserialize(*args, **kwargs)
        Tree._deserialize = counting_deserialize
        t = self.get_tree(tree_sha)
        self.assertEqual(t.items()[0], (b'a', 33188, a_sha))
        self.assertEqual(t.items()[1], (b'b', 33188, b_sha))
        self.assertEqual(self.deserialize_count, 1)

    def test_read_tag_from_file(self):
        t = self.get_tag(tag_sha)
        self.assertEqual(t.object, (Commit, b'51b668fd5bf7061b7d6fa525f88803e6cfadaa51'))
        self.assertEqual(t.name, b'signed')
        self.assertEqual(t.tagger, b'Ali Sabil <ali.sabil@gmail.com>')
        self.assertEqual(t.tag_time, 1231203091)
        self.assertEqual(t.message, b'This is a signed tag\n')
        self.assertEqual(t.signature, b'-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.9 (GNU/Linux)\n\niEYEABECAAYFAkliqx8ACgkQqSMmLy9u/kcx5ACfakZ9NnPl02tOyYP6pkBoEkU1\n5EcAn0UFgokaSvS371Ym/4W9iJj6vh3h\n=ql7y\n-----END PGP SIGNATURE-----\n')

    def test_read_commit_from_file(self):
        sha = b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e'
        c = self.commit(sha)
        self.assertEqual(c.tree, tree_sha)
        self.assertEqual(c.parents, [b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'])
        self.assertEqual(c.author, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.committer, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.commit_time, 1174759230)
        self.assertEqual(c.commit_timezone, 0)
        self.assertEqual(c.author_timezone, 0)
        self.assertEqual(c.message, b'Test commit\n')

    def test_read_commit_no_parents(self):
        sha = b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'
        c = self.commit(sha)
        self.assertEqual(c.tree, b'90182552c4a85a45ec2a835cadc3451bebdfe870')
        self.assertEqual(c.parents, [])
        self.assertEqual(c.author, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.committer, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.commit_time, 1174758034)
        self.assertEqual(c.commit_timezone, 0)
        self.assertEqual(c.author_timezone, 0)
        self.assertEqual(c.message, b'Test commit\n')

    def test_read_commit_two_parents(self):
        sha = b'5dac377bdded4c9aeb8dff595f0faeebcc8498cc'
        c = self.commit(sha)
        self.assertEqual(c.tree, b'd80c186a03f423a81b39df39dc87fd269736ca86')
        self.assertEqual(c.parents, [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'])
        self.assertEqual(c.author, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.committer, b'James Westby <jw+debian@jameswestby.net>')
        self.assertEqual(c.commit_time, 1174773719)
        self.assertEqual(c.commit_timezone, 0)
        self.assertEqual(c.author_timezone, 0)
        self.assertEqual(c.message, b'Merge ../b\n')

    def test_stub_sha(self):
        sha = b'5' * 40
        c = make_commit(id=sha, message=b'foo')
        self.assertIsInstance(c, Commit)
        self.assertEqual(sha, c.id)
        self.assertNotEqual(sha, c.sha())