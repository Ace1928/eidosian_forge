from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestNotesDisplay(TestCase):

    def test_noteonly(self):
        c = commands.NoteModifyCommand(b'foo', b'A basic note')
        self.assertEqual(b'N inline :foo\ndata 12\nA basic note', bytes(c))

    def test_notecommit(self):
        committer = (b'Ed Mund', b'ed@example.org', 1234565432, 0)
        commits = [commands.CommitCommand(ref=b'refs/heads/master', mark=b'1', author=committer, committer=committer, message=b'test\n', from_=None, merges=[], file_iter=[commands.FileModifyCommand(b'bar', 33188, None, b'')]), commands.CommitCommand(ref=b'refs/notes/commits', mark=None, author=None, committer=committer, message=b"Notes added by 'git notes add'\n", from_=None, merges=[], file_iter=[commands.NoteModifyCommand(b'1', b'Test note\n')]), commands.CommitCommand(ref=b'refs/notes/test', mark=None, author=None, committer=committer, message=b"Notes added by 'git notes add'\n", from_=None, merges=[], file_iter=[commands.NoteModifyCommand(b'1', b'Test test\n')])]
        self.assertEqual(b"commit refs/heads/master\nmark :1\nauthor Ed Mund <ed@example.org> 1234565432 +0000\ncommitter Ed Mund <ed@example.org> 1234565432 +0000\ndata 5\ntest\n\nM 644 inline bar\ndata 0\ncommit refs/notes/commits\ncommitter Ed Mund <ed@example.org> 1234565432 +0000\ndata 31\nNotes added by 'git notes add'\n\nN inline :1\ndata 10\nTest note\ncommit refs/notes/test\ncommitter Ed Mund <ed@example.org> 1234565432 +0000\ndata 31\nNotes added by 'git notes add'\n\nN inline :1\ndata 10\nTest test\n", b''.join([bytes(s) for s in commits]))