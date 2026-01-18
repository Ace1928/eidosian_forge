from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
class TestRevProps(TestCaseWithRepository):

    def test_simple_revprops(self):
        """Simple revision properties"""
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        b.nick = 'Nicholas'
        if b.repository._format.supports_custom_revision_properties:
            props = {'flavor': 'choc-mint', 'condiment': 'orange\n  mint\n\tcandy', 'empty': '', 'non_ascii': 'µ'}
        else:
            props = {}
        rev1 = wt.commit(message='initial null commit', revprops=props, allow_pointless=True)
        rev = b.repository.get_revision(rev1)
        if b.repository._format.supports_custom_revision_properties:
            self.assertTrue('flavor' in rev.properties)
            self.assertEqual(rev.properties['flavor'], 'choc-mint')
            expected_revprops = {'condiment': 'orange\n  mint\n\tcandy', 'empty': '', 'flavor': 'choc-mint', 'non_ascii': 'µ'}
        else:
            expected_revprops = {}
        if b.repository._format.supports_storing_branch_nick:
            expected_revprops['branch-nick'] = 'Nicholas'
        for name, value in expected_revprops.items():
            self.assertEqual(rev.properties[name], value)

    def test_invalid_revprops(self):
        """Invalid revision properties"""
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        if not b.repository._format.supports_custom_revision_properties:
            raise TestNotApplicable('format does not support custom revision properties')
        self.assertRaises(ValueError, wt.commit, message='invalid', revprops={'what a silly property': 'fine'})
        self.assertRaises(ValueError, wt.commit, message='invalid', revprops=dict(number=13))