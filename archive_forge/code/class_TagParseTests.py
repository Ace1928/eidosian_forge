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
class TagParseTests(ShaFileCheckTests):

    def make_tag_lines(self, object_sha=b'a38d6181ff27824c79fc7df825164a212eff6a3f', object_type_name=b'commit', name=b'v2.6.22-rc7', tagger=default_tagger, message=default_message):
        lines = []
        if object_sha is not None:
            lines.append(b'object ' + object_sha)
        if object_type_name is not None:
            lines.append(b'type ' + object_type_name)
        if name is not None:
            lines.append(b'tag ' + name)
        if tagger is not None:
            lines.append(b'tagger ' + tagger)
        if message is not None:
            lines.append(b'')
            lines.append(message)
        return lines

    def make_tag_text(self, **kwargs):
        return b'\n'.join(self.make_tag_lines(**kwargs))

    def test_parse(self):
        x = Tag()
        x.set_raw_string(self.make_tag_text())
        self.assertEqual(b'Linus Torvalds <torvalds@woody.linux-foundation.org>', x.tagger)
        self.assertEqual(b'v2.6.22-rc7', x.name)
        object_type, object_sha = x.object
        self.assertEqual(b'a38d6181ff27824c79fc7df825164a212eff6a3f', object_sha)
        self.assertEqual(Commit, object_type)
        self.assertEqual(datetime.datetime.utcfromtimestamp(x.tag_time), datetime.datetime(2007, 7, 1, 19, 54, 34))
        self.assertEqual(-25200, x.tag_timezone)

    def test_parse_no_tagger(self):
        x = Tag()
        x.set_raw_string(self.make_tag_text(tagger=None))
        self.assertEqual(None, x.tagger)
        self.assertEqual(b'v2.6.22-rc7', x.name)
        self.assertEqual(None, x.tag_time)

    def test_parse_no_message(self):
        x = Tag()
        x.set_raw_string(self.make_tag_text(message=None))
        self.assertEqual(None, x.message)
        self.assertEqual(b'Linus Torvalds <torvalds@woody.linux-foundation.org>', x.tagger)
        self.assertEqual(datetime.datetime.utcfromtimestamp(x.tag_time), datetime.datetime(2007, 7, 1, 19, 54, 34))
        self.assertEqual(-25200, x.tag_timezone)
        self.assertEqual(b'v2.6.22-rc7', x.name)

    def test_check(self):
        self.assertCheckSucceeds(Tag, self.make_tag_text())
        self.assertCheckFails(Tag, self.make_tag_text(object_sha=None))
        self.assertCheckFails(Tag, self.make_tag_text(object_type_name=None))
        self.assertCheckFails(Tag, self.make_tag_text(name=None))
        self.assertCheckFails(Tag, self.make_tag_text(name=b''))
        self.assertCheckFails(Tag, self.make_tag_text(object_type_name=b'foobar'))
        self.assertCheckFails(Tag, self.make_tag_text(tagger=b'some guy without an email address 1183319674 -0700'))
        self.assertCheckFails(Tag, self.make_tag_text(tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org> Sun 7 Jul 2007 12:54:34 +0700'))
        self.assertCheckFails(Tag, self.make_tag_text(object_sha=b'xxx'))

    def test_check_tag_with_unparseable_field(self):
        self.assertCheckFails(Tag, self.make_tag_text(tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org> 423423+0000'))

    def test_check_tag_with_overflow_time(self):
        """Date with overflow should raise an ObjectFormatException when checked."""
        author = f'Some Dude <some@dude.org> {MAX_TIME + 1} +0000'
        tag = Tag.from_string(self.make_tag_text(tagger=author.encode()))
        with self.assertRaises(ObjectFormatException):
            tag.check()

    def test_check_duplicates(self):
        for i in range(4):
            lines = self.make_tag_lines()
            lines.insert(i, lines[i])
            self.assertCheckFails(Tag, b'\n'.join(lines))

    def test_check_order(self):
        lines = self.make_tag_lines()
        headers = lines[:4]
        rest = lines[4:]
        for perm in permutations(headers):
            perm = list(perm)
            text = b'\n'.join(perm + rest)
            if perm == headers:
                self.assertCheckSucceeds(Tag, text)
            else:
                self.assertCheckFails(Tag, text)

    def test_tree_copy_after_update(self):
        """Check Tree.id is correctly updated when the tree is copied after updated."""
        shas = []
        tree = Tree()
        shas.append(tree.id)
        tree.add(b'data', 420, Blob().id)
        copied = tree.copy()
        shas.append(tree.id)
        shas.append(copied.id)
        self.assertNotIn(shas[0], shas[1:])
        self.assertEqual(shas[1], shas[2])