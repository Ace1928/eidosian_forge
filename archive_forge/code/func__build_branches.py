import os
from breezy.tests import TestCaseWithTransport
def _build_branches(self):
    a_wt = self.make_branch_and_tree('A')
    self.build_tree_contents([('A/foo', b'1111\n')])
    a_wt.add('foo')
    a_wt.commit('added foo', rev_id=b'A1')
    b_wt = a_wt.controldir.sprout('B').open_workingtree()
    self.build_tree_contents([('B/foo', b'1111\n22\n')])
    b_wt.commit('modified B/foo', rev_id=b'B1')
    self.build_tree_contents([('A/foo', b'000\n1111\n')])
    a_wt.commit('modified A/foo', rev_id=b'A2')
    a_wt.merge_from_branch(b_wt.branch, b_wt.last_revision(), b_wt.branch.get_rev_id(1))
    a_wt.commit('merged B into A', rev_id=b'A3')
    return (a_wt, b_wt)