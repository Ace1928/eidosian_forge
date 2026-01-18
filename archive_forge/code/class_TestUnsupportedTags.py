from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
class TestUnsupportedTags(per_branch.TestCaseWithBranch):
    """Formats that don't support tags should give reasonable errors."""

    def setUp(self):
        super().setUp()
        branch = self.make_branch('probe')
        if branch._format.supports_tags():
            raise tests.TestSkipped('Format %s declares that tags are supported' % branch._format)

    def test_tag_methods_raise(self):
        b = self.make_branch('b')
        self.assertRaises(errors.TagsNotSupported, b.tags.set_tag, 'foo', 'bar')
        self.assertRaises(errors.TagsNotSupported, b.tags.lookup_tag, 'foo')
        self.assertRaises(errors.TagsNotSupported, b.tags.set_tag, 'foo', 'bar')
        self.assertRaises(errors.TagsNotSupported, b.tags.delete_tag, 'foo')

    def test_merge_empty_tags(self):
        b1 = self.make_branch('b1')
        b2 = self.make_branch('b2')
        b1.tags.merge_to(b2.tags)