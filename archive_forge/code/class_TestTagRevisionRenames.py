from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
class TestTagRevisionRenames(TestCaseWithTransport):

    def make_branch_supporting_tags(self, relpath):
        return self.make_branch(relpath, format='dirstate-tags')

    def test_simple(self):
        store = self.make_branch_supporting_tags('a').tags
        store.set_tag('foo', b'myoldrevid')
        store.rename_revisions({b'myoldrevid': b'mynewrevid'})
        self.assertEqual({'foo': b'mynewrevid'}, store.get_tag_dict())

    def test_unknown_ignored(self):
        store = self.make_branch_supporting_tags('a').tags
        store.set_tag('foo', b'myoldrevid')
        store.rename_revisions({b'anotherrevid': b'mynewrevid'})
        self.assertEqual({'foo': b'myoldrevid'}, store.get_tag_dict())