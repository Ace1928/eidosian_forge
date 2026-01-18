from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
class TestGitBranchBuilder(tests.TestCase):

    def test__create_blob(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        self.assertEqual(1, builder._create_blob(b'foo\nbar\n'))
        self.assertEqualDiff(b'blob\nmark :1\ndata 8\nfoo\nbar\n\n', stream.getvalue())

    def test_set_file(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('foobar', b'foo\nbar\n', False)
        self.assertEqualDiff(b'blob\nmark :1\ndata 8\nfoo\nbar\n\n', stream.getvalue())
        self.assertEqual([b'M 100644 :1 foobar\n'], builder.commit_info)

    def test_set_file_unicode(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('fµ/bar', b'contents\nbar\n', False)
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\n', stream.getvalue())
        self.assertEqual([b'M 100644 :1 f\xc2\xb5/bar\n'], builder.commit_info)

    def test_set_file_newline(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('foo\nbar', b'contents\nbar\n', False)
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\n', stream.getvalue())
        self.assertEqual([b'M 100644 :1 "foo\\nbar"\n'], builder.commit_info)

    def test_set_file_executable(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('fµ/bar', b'contents\nbar\n', True)
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\n', stream.getvalue())
        self.assertEqual([b'M 100755 :1 f\xc2\xb5/bar\n'], builder.commit_info)

    def test_set_symlink(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_symlink('fµ/bar', b'link/contents')
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\nlink/contents\n', stream.getvalue())
        self.assertEqual([b'M 120000 :1 f\xc2\xb5/bar\n'], builder.commit_info)

    def test_set_symlink_newline(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_symlink('foo\nbar', 'link/contents')
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\nlink/contents\n', stream.getvalue())
        self.assertEqual([b'M 120000 :1 "foo\\nbar"\n'], builder.commit_info)

    def test_delete_entry(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.delete_entry('path/to/fµ')
        self.assertEqual([b'D path/to/f\xc2\xb5\n'], builder.commit_info)

    def test_delete_entry_newline(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.delete_entry('path/to/foo\nbar')
        self.assertEqual([b'D "path/to/foo\\nbar"\n'], builder.commit_info)

    def test_encode_path(self):
        encode = tests.GitBranchBuilder._encode_path
        self.assertEqual(encode('fµ'), b'f\xc2\xb5')
        self.assertEqual(encode('"foo'), b'"\\"foo"')
        self.assertEqual(encode('fo\no'), b'"fo\\no"')
        self.assertEqual(encode('fo\\o\nbar'), b'"fo\\\\o\\nbar"')
        self.assertEqual(encode('fo"o"\nbar'), b'"fo\\"o\\"\\nbar"')
        self.assertEqual(encode('foo\r\nbar'), b'"foo\r\\nbar"')

    def test_add_and_commit(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('fµ/bar', b'contents\nbar\n', False)
        self.assertEqual(b'2', builder.commit(b'Joe Foo <joe@foo.com>', 'committing fµ/bar', timestamp=1194586400, timezone=b'+0100'))
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1194586400 +0100\ndata 18\ncommitting f\xc2\xb5/bar\nM 100644 :1 f\xc2\xb5/bar\n\n', stream.getvalue())

    def test_commit_base(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('foo', b'contents\nfoo\n', False)
        r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'first', timestamp=1194586400)
        r2 = builder.commit(b'Joe Foo <joe@foo.com>', 'second', timestamp=1194586405)
        r3 = builder.commit(b'Joe Foo <joe@foo.com>', 'third', timestamp=1194586410, base=r1)
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nfoo\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1194586400 +0000\ndata 5\nfirst\nM 100644 :1 foo\n\ncommit refs/heads/master\nmark :3\ncommitter Joe Foo <joe@foo.com> 1194586405 +0000\ndata 6\nsecond\n\ncommit refs/heads/master\nmark :4\ncommitter Joe Foo <joe@foo.com> 1194586410 +0000\ndata 5\nthird\nfrom :2\n\n', stream.getvalue())

    def test_commit_merge(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.set_file('foo', b'contents\nfoo\n', False)
        r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'first', timestamp=1194586400)
        r2 = builder.commit(b'Joe Foo <joe@foo.com>', 'second', timestamp=1194586405)
        r3 = builder.commit(b'Joe Foo <joe@foo.com>', 'third', timestamp=1194586410, base=r1)
        r4 = builder.commit(b'Joe Foo <joe@foo.com>', 'Merge', timestamp=1194586415, merge=[r2])
        self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nfoo\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1194586400 +0000\ndata 5\nfirst\nM 100644 :1 foo\n\ncommit refs/heads/master\nmark :3\ncommitter Joe Foo <joe@foo.com> 1194586405 +0000\ndata 6\nsecond\n\ncommit refs/heads/master\nmark :4\ncommitter Joe Foo <joe@foo.com> 1194586410 +0000\ndata 5\nthird\nfrom :2\n\ncommit refs/heads/master\nmark :5\ncommitter Joe Foo <joe@foo.com> 1194586415 +0000\ndata 5\nMerge\nmerge :3\n\n', stream.getvalue())

    def test_auto_timestamp(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.commit(b'Joe Foo <joe@foo.com>', 'message')
        self.assertContainsRe(stream.getvalue(), b'committer Joe Foo <joe@foo\\.com> \\d+ \\+0000')

    def test_reset(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.reset()
        self.assertEqualDiff(b'reset refs/heads/master\n\n', stream.getvalue())

    def test_reset_named_ref(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.reset(b'refs/heads/branch')
        self.assertEqualDiff(b'reset refs/heads/branch\n\n', stream.getvalue())

    def test_reset_revision(self):
        stream = BytesIO()
        builder = tests.GitBranchBuilder(stream)
        builder.reset(mark=b'123')
        self.assertEqualDiff(b'reset refs/heads/master\nfrom :123\n\n', stream.getvalue())