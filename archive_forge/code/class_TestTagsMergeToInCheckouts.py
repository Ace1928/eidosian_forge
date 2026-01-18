from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
class TestTagsMergeToInCheckouts(per_branch.TestCaseWithBranch):
    """Tests for checkout.branch.tags.merge_to.

    In particular this exercises variations in tag conflicts in the master
    branch and/or the checkout (child).  It may seem strange to have different
    tags in the child and master, but 'bzr merge' intentionally updates the
    child and not the master (instead the next 'bzr commit', if the user
    decides to commit, will update the master).  Also, merge_to in bzr < 2.3
    didn't propagate changes to the master, and current bzr versions may find
    themselves operating on checkouts touched by older bzrs

    So we need to make sure bzr copes gracefully with differing tags in the
    master versus the child.

    See also <https://bugs.launchpad.net/bzr/+bug/603395>.
    """

    def setUp(self):
        super().setUp()
        branch1 = self.make_branch('tags-probe')
        if not branch1._format.supports_tags():
            raise tests.TestSkipped("format %s doesn't support tags" % branch1._format)
        branch2 = self.make_branch('bind-probe')
        try:
            branch2.bind(branch1)
        except branch.BindingUnsupported:
            raise tests.TestNotApplicable("format %s doesn't support bound branches" % branch2._format)

    def test_merge_to_propagates_tags(self):
        """merge_to(child) also merges tags to the master."""
        master = self.make_branch('master')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        other.tags.merge_to(child.tags)
        self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))

    def test_ignore_master_disables_tag_propagation(self):
        """merge_to(child, ignore_master=True) does not merge tags to the
        master.
        """
        master = self.make_branch('master')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        other.tags.merge_to(child.tags, ignore_master=True)
        self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
        self.assertRaises(errors.NoSuchTag, master.tags.lookup_tag, 'foo')

    def test_merge_to_overwrite_conflict_in_master(self):
        """merge_to(child, overwrite=True) overwrites any conflicting tags in
        the master.
        """
        master = self.make_branch('master')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        master.tags.set_tag('foo', b'rev-2')
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags, overwrite=True)
        self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
        self.assertEqual({'foo': b'rev-1'}, tag_updates)
        self.assertLength(0, tag_conflicts)

    def test_merge_to_overwrite_conflict_in_child_and_master(self):
        """merge_to(child, overwrite=True) overwrites any conflicting tags in
        both the child and the master.
        """
        master = self.make_branch('master')
        master.tags.set_tag('foo', b'rev-2')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags, overwrite=True)
        self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
        self.assertEqual({'foo': b'rev-1'}, tag_updates)
        self.assertLength(0, tag_conflicts)

    def test_merge_to_conflict_in_child_only(self):
        """When new_tags.merge_to(child.tags) conflicts with the child but not
        the master, a conflict is reported and the child receives the new tag.
        """
        master = self.make_branch('master')
        master.tags.set_tag('foo', b'rev-2')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        master.tags.delete_tag('foo')
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags)
        self.assertEqual(b'rev-2', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
        self.assertEqual([('foo', b'rev-1', b'rev-2')], list(tag_conflicts))
        self.assertEqual({'foo': b'rev-1'}, tag_updates)

    def test_merge_to_conflict_in_master_only(self):
        """When new_tags.merge_to(child.tags) conflicts with the master but not
        the child, a conflict is reported and the child receives the new tag.
        """
        master = self.make_branch('master')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        master.tags.set_tag('foo', b'rev-2')
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags)
        self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-2', master.tags.lookup_tag('foo'))
        self.assertEqual({'foo': b'rev-1'}, tag_updates)
        self.assertEqual([('foo', b'rev-1', b'rev-2')], list(tag_conflicts))

    def test_merge_to_same_conflict_in_master_and_child(self):
        """When new_tags.merge_to(child.tags) conflicts the same way with the
        master and the child a single conflict is reported.
        """
        master = self.make_branch('master')
        master.tags.set_tag('foo', b'rev-2')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags)
        self.assertEqual(b'rev-2', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-2', master.tags.lookup_tag('foo'))
        self.assertEqual({}, tag_updates)
        self.assertEqual([('foo', b'rev-1', b'rev-2')], list(tag_conflicts))

    def test_merge_to_different_conflict_in_master_and_child(self):
        """When new_tags.merge_to(child.tags) conflicts differently in the
        master and the child both conflicts are reported.
        """
        master = self.make_branch('master')
        master.tags.set_tag('foo', b'rev-2')
        other = self.make_branch('other')
        other.tags.set_tag('foo', b'rev-1')
        child = self.make_branch('child')
        child.bind(master)
        child.update()
        child.tags._set_tag_dict({'foo': b'rev-3'})
        tag_updates, tag_conflicts = other.tags.merge_to(child.tags)
        self.assertEqual(b'rev-3', child.tags.lookup_tag('foo'))
        self.assertEqual(b'rev-2', master.tags.lookup_tag('foo'))
        self.assertEqual({}, tag_updates)
        self.assertEqual([('foo', b'rev-1', b'rev-2'), ('foo', b'rev-1', b'rev-3')], sorted(tag_conflicts))