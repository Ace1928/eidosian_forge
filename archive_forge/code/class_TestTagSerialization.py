from breezy.bzr.tag import BasicTags
from breezy.tests import TestCase
class TestTagSerialization(TestCase):

    def test_tag_serialization(self):
        """Test the precise representation of tag dicts."""
        store = BasicTags(branch=None)
        td = dict(stable=b'stable-revid', boring=b'boring-revid')
        packed = store._serialize_tag_dict(td)
        expected = b'd6:boring12:boring-revid6:stable12:stable-revide'
        self.assertEqualDiff(packed, expected)
        self.assertEqual(store._deserialize_tag_dict(packed), td)