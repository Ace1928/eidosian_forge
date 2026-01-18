from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
class WriteCommitPatchTests(TestCase):

    def test_simple_bytesio(self):
        f = BytesIO()
        c = Commit()
        c.committer = c.author = b'Jelmer <jelmer@samba.org>'
        c.commit_time = c.author_time = 1271350201
        c.commit_timezone = c.author_timezone = 0
        c.message = b'This is the first line\nAnd this is the second line.\n'
        c.tree = Tree().id
        write_commit_patch(f, c, b'CONTENTS', (1, 1), version='custom')
        f.seek(0)
        lines = f.readlines()
        self.assertTrue(lines[0].startswith(b'From 0b0d34d1b5b596c928adc9a727a4b9e03d025298'))
        self.assertEqual(lines[1], b'From: Jelmer <jelmer@samba.org>\n')
        self.assertTrue(lines[2].startswith(b'Date: '))
        self.assertEqual([b'Subject: [PATCH 1/1] This is the first line\n', b'And this is the second line.\n', b'\n', b'\n', b'---\n'], lines[3:8])
        self.assertEqual([b'CONTENTS-- \n', b'custom\n'], lines[-2:])
        if len(lines) >= 12:
            self.assertEqual(lines[8], b' 0 files changed\n')