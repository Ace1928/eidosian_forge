from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
class MemoryTagsTests(TestCase):

    def setUp(self):
        super().setUp()
        self.tags = MemoryTags({})

    def test_set_tag(self):
        self.tags.set_tag('foo', b'revid1')
        self.assertEqual({'foo': b'revid1'}, self.tags.get_tag_dict())

    def test_reverse_tag_dict(self):
        self.tags.set_tag('foo', b'revid1')
        self.tags.set_tag('bar', b'revid2')
        self.tags.set_tag('blah', b'revid1')
        self.assertEqual({b'revid1': {'foo', 'blah'}, b'revid2': {'bar'}}, self.tags.get_reverse_tag_dict())

    def test_lookup_tag(self):
        self.tags.set_tag('foo', b'revid1')
        self.assertEqual(b'revid1', self.tags.lookup_tag('foo'))
        self.assertRaises(errors.NoSuchTag, self.tags.lookup_tag, 'bar')

    def test_delete_tag(self):
        self.tags.set_tag('foo', b'revid1')
        self.assertEqual(b'revid1', self.tags.lookup_tag('foo'))
        self.tags.delete_tag('foo')
        self.assertRaises(errors.NoSuchTag, self.tags.lookup_tag, 'foo')
        self.assertRaises(errors.NoSuchTag, self.tags.delete_tag, 'foo')

    def test_has_tag(self):
        self.tags.set_tag('foo', b'revid1')
        self.assertTrue(self.tags.has_tag('foo'))
        self.assertFalse(self.tags.has_tag('bar'))

    def test_rename_revisions(self):
        self.tags.set_tag('foo', b'revid1')
        self.assertEqual({'foo': b'revid1'}, self.tags.get_tag_dict())
        self.tags.rename_revisions({b'revid1': b'revid2'})
        self.assertEqual({'foo': b'revid2'}, self.tags.get_tag_dict())

    def test_merge_to(self):
        other_tags = MemoryTags({})
        other_tags.set_tag('tag-1', b'x')
        self.tags.set_tag('tag-2', b'y')
        other_tags.merge_to(self.tags)
        self.assertEqual(b'x', self.tags.lookup_tag('tag-1'))
        self.assertEqual(b'y', self.tags.lookup_tag('tag-2'))
        self.assertRaises(errors.NoSuchTag, other_tags.lookup_tag, 'tag-2')
        other_tags.set_tag('tag-2', b'z')
        updates, conflicts = other_tags.merge_to(self.tags)
        self.assertEqual({}, updates)
        self.assertEqual(list(conflicts), [('tag-2', b'z', b'y')])
        self.assertEqual(b'y', self.tags.lookup_tag('tag-2'))
        updates, conflicts = other_tags.merge_to(self.tags, overwrite=True)
        self.assertEqual(list(conflicts), [])
        self.assertEqual({'tag-2': b'z'}, updates)
        self.assertEqual(b'z', self.tags.lookup_tag('tag-2'))