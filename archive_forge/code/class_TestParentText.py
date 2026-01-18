from unittest import TestCase
import patiencediff
from .. import multiparent, tests
class TestParentText(TestCase):

    def test_eq(self):
        self.assertEqual(multiparent.ParentText(1, 2, 3, 4), multiparent.ParentText(1, 2, 3, 4))
        self.assertFalse(multiparent.ParentText(1, 2, 3, 4) == multiparent.ParentText(2, 2, 3, 4))
        self.assertFalse(multiparent.ParentText(1, 2, 3, 4) == Mock(parent=1, parent_pos=2, child_pos=3, num_lines=4))

    def test_to_patch(self):
        self.assertEqual([b'c 0 1 2 3\n'], list(multiparent.ParentText(0, 1, 2, 3).to_patch()))