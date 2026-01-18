from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
class DisabledTagsTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        branch = self.make_branch('.')
        self.tags = DisabledTags(branch)

    def test_set_tag(self):
        self.assertRaises(errors.TagsNotSupported, self.tags.set_tag)

    def test_get_reverse_tag_dict(self):
        self.assertEqual(self.tags.get_reverse_tag_dict(), {})