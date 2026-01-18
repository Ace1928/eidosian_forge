from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestCommitDisplay(TestCase):

    def test_commit(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, None)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))

    def test_commit_unicode_committer(self):
        name = u'Ľórém Ípšúm'
        commit_utf8 = utf8_bytes_string(u'commit refs/heads/master\nmark :bbb\ncommitter %s <test@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa' % (name,))
        committer = (name, b'test@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, None)
        self.assertEqual(commit_utf8, bytes(c))

    def test_commit_no_mark(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', None, None, committer, b'release v1.0', b':aaa', None, None)
        self.assertEqual(b'commit refs/heads/master\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))

    def test_commit_no_from(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', None, None, None)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0', bytes(c))

    def test_commit_with_author(self):
        author = (b'Sue Wong', b'sue@example.com', 1234565432, -6 * 3600)
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', author, committer, b'release v1.0', b':aaa', None, None)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\nauthor Sue Wong <sue@example.com> 1234565432 -0600\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))

    def test_commit_with_merges(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'ddd', None, committer, b'release v1.0', b':aaa', [b':bbb', b':ccc'], None)
        self.assertEqual(b'commit refs/heads/master\nmark :ddd\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nmerge :bbb\nmerge :ccc', bytes(c))

    def test_commit_with_filecommands(self):
        file_cmds = iter([commands.FileDeleteCommand(b'readme.txt'), commands.FileModifyCommand(b'NEWS', 33188, None, b'blah blah blah')])
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, file_cmds)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nD readme.txt\nM 644 inline NEWS\ndata 14\nblah blah blah', bytes(c))

    def test_commit_with_more_authors(self):
        author = (b'Sue Wong', b'sue@example.com', 1234565432, -6 * 3600)
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        more_authors = [(b'Al Smith', b'al@example.com', 1234565432, -6 * 3600), (b'Bill Jones', b'bill@example.com', 1234565432, -6 * 3600)]
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', author, committer, b'release v1.0', b':aaa', None, None, more_authors=more_authors)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\nauthor Sue Wong <sue@example.com> 1234565432 -0600\nauthor Al Smith <al@example.com> 1234565432 -0600\nauthor Bill Jones <bill@example.com> 1234565432 -0600\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))

    def test_commit_with_properties(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        properties = {u'greeting': u'hello', u'planet': u'world'}
        c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, None, properties=properties)
        self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nproperty greeting 5 hello\nproperty planet 5 world', bytes(c))

    def test_commit_with_int_mark(self):
        committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        properties = {u'greeting': u'hello', u'planet': u'world'}
        c = commands.CommitCommand(b'refs/heads/master', 123, None, committer, b'release v1.0', b':aaa', None, None, properties=properties)
        self.assertEqual(b'commit refs/heads/master\nmark :123\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nproperty greeting 5 hello\nproperty planet 5 world', bytes(c))