from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
class TestCalcParent_name(tests.TestCase):
    """Tests for calc_parent_name."""

    def test_dotted_member(self):
        self.assertEqual(('mod_name', 'attr1', 'attr2'), calc_parent_name('mod_name', 'attr1.attr2'))

    def test_undotted_member(self):
        self.assertEqual(('mod_name', None, 'attr1'), calc_parent_name('mod_name', 'attr1'))

    def test_dotted_module_no_member(self):
        self.assertEqual(('mod', None, 'sub_mod'), calc_parent_name('mod.sub_mod'))

    def test_undotted_module_no_member(self):
        err = self.assertRaises(AssertionError, calc_parent_name, 'mod_name')
        self.assertEqual("No parent object for top-level module 'mod_name'", err.args[0])