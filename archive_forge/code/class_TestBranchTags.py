from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
class TestBranchTags(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        branch = self.make_branch('probe')
        if not branch._format.supports_tags():
            raise tests.TestSkipped("format %s doesn't support tags" % branch._format)

    def make_branch_with_revisions(self, relpath, revisions):
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        for revid in revisions:
            builder.build_commit(rev_id=revid)
        builder.finish_series()
        return builder.get_branch()

    def make_branch_with_revision_tuple(self, relpath, count):
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        revids = [builder.build_commit() for i in range(count)]
        builder.finish_series()
        return (builder.get_branch(), revids)

    def test_tags_initially_empty(self):
        b = self.make_branch('b')
        tags = b.tags.get_tag_dict()
        self.assertEqual(tags, {})

    def test_make_and_lookup_tag(self):
        b, [target_revid1, target_revid2] = self.make_branch_with_revision_tuple('b', 2)
        b.tags.set_tag('tag-name', target_revid1)
        b.tags.set_tag('other-name', target_revid2)
        b = branch.Branch.open('b')
        self.assertEqual(b.tags.get_tag_dict(), {'tag-name': target_revid1, 'other-name': target_revid2})
        result = b.tags.lookup_tag('tag-name')
        self.assertEqual(result, target_revid1)
        self.assertTrue(b.tags.has_tag('tag-name'))
        self.assertFalse(b.tags.has_tag('imaginary'))

    def test_reverse_tag_dict(self):
        b, [target_revid1, target_revid2] = self.make_branch_with_revision_tuple('b', 2)
        b.tags.set_tag('tag-name', target_revid1)
        b.tags.set_tag('other-name', target_revid2)
        b = branch.Branch.open('b')
        self.assertEqual(dict(b.tags.get_reverse_tag_dict()), {target_revid1: {'tag-name'}, target_revid2: {'other-name'}})

    def test_ghost_tag(self):
        b = self.make_branch('b')
        if not b._format.supports_tags_referencing_ghosts():
            self.assertRaises(errors.GhostTagsNotSupported, b.tags.set_tag, 'ghost', b'idontexist')
        else:
            b.tags.set_tag('ghost', b'idontexist')
            self.assertEqual(b'idontexist', b.tags.lookup_tag('ghost'))

    def test_no_such_tag(self):
        b = self.make_branch('b')
        try:
            b.tags.lookup_tag('bosko')
        except errors.NoSuchTag as e:
            self.assertEqual(e.tag_name, 'bosko')
            self.assertEqual(str(e), 'No such tag: bosko')
        else:
            self.fail("didn't get expected exception")

    def test_merge_tags(self):
        b1, [revid, revid1] = self.make_branch_with_revision_tuple('b1', 2)
        w2 = b1.controldir.sprout('b2', revision_id=revid).open_workingtree()
        revid2 = w2.commit('revision 2')
        b2 = w2.branch
        b1.tags.set_tag('tagname', revid)
        b1.tags.merge_to(b2.tags)
        self.assertEqual(b2.tags.lookup_tag('tagname'), revid)
        b2.tags.set_tag('in-destination', revid)
        updates, conflicts = b1.tags.merge_to(b2.tags)
        self.assertEqual(list(conflicts), [])
        self.assertEqual(updates, {})
        self.assertEqual(b2.tags.lookup_tag('in-destination'), revid)
        b1.tags.set_tag('conflicts', revid1)
        b2.tags.set_tag('conflicts', revid2)
        updates, conflicts = b1.tags.merge_to(b2.tags)
        self.assertEqual(list(conflicts), [('conflicts', revid1, revid2)])
        self.assertEqual(updates, {})
        self.assertEqual(b2.tags.lookup_tag('conflicts'), revid2)

    def test_merge_tags_selector(self):
        b1, [revid, revid1] = self.make_branch_with_revision_tuple('b1', 2)
        w2 = b1.controldir.sprout('b2', revision_id=revid).open_workingtree()
        revid2 = w2.commit('revision 2')
        b2 = w2.branch
        b1.tags.set_tag('tag1', revid)
        b1.tags.set_tag('tag2', revid2)
        updates, conflicts = b1.tags.merge_to(b2.tags, selector=lambda x: x == 'tag1')
        self.assertEqual({'tag1': revid}, updates)
        self.assertEqual(set(), set(conflicts))
        self.assertEqual(b2.tags.lookup_tag('tag1'), revid)
        self.assertRaises(errors.NoSuchTag, b2.tags.lookup_tag, 'tag2')

    def test_unicode_tag(self):
        tag_name = 'ば'
        b1, [revid] = self.make_branch_with_revision_tuple('b', 1)
        b1.tags.set_tag(tag_name, revid)
        self.assertEqual(b1.tags.lookup_tag(tag_name), revid)

    def test_delete_tag(self):
        tag_name = 'α'
        b, [revid] = self.make_branch_with_revision_tuple('b', 1)
        b.tags.set_tag(tag_name, revid)
        b.tags.delete_tag(tag_name)
        self.assertRaises(errors.NoSuchTag, b.tags.lookup_tag, tag_name)
        self.assertEqual(b.tags.get_tag_dict(), {})
        self.assertRaises(errors.NoSuchTag, b.tags.delete_tag, tag_name)
        self.assertRaises(errors.NoSuchTag, b.tags.delete_tag, tag_name + '2')

    def test_merge_empty_tags(self):
        b1 = self.make_branch('b1')
        b2 = self.make_branch('b2')
        b1.tags.merge_to(b2.tags)

    def test_read_lock_caches_tags(self):
        """Tags are read from a branch only once during a read-lock."""
        b1, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
        b1.tags.set_tag('one', rev1)
        b2 = controldir.ControlDir.open('b').open_branch()
        b1.lock_read()
        self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
        b2.tags.set_tag('one', rev2)
        b2.tags.set_tag('two', rev3)
        self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
        b1.unlock()
        self.assertEqual({'one': rev2, 'two': rev3}, b1.tags.get_tag_dict())

    def test_unlocked_does_not_cache_tags(self):
        """Unlocked branches do not cache tags."""
        b1, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
        b1.tags.set_tag('one', rev1)
        b2 = b1.controldir.open_branch()
        self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
        b2.tags.set_tag('one', rev2)
        b2.tags.set_tag('two', rev3)
        self.assertEqual({'one': rev2, 'two': rev3}, b1.tags.get_tag_dict())

    def test_cached_tag_dict_not_accidentally_mutable(self):
        """When there's a cached version of the tags, b.tags.get_tag_dict
        returns a copy of the cached data so that callers cannot accidentally
        corrupt the cache.
        """
        b, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
        b.tags.set_tag('one', rev1)
        self.addCleanup(b.lock_read().unlock)
        tags_dict = b.tags.get_tag_dict()
        tags_dict['two'] = rev2
        tags_dict = b.tags.get_tag_dict()
        tags_dict['three'] = rev3
        self.assertEqual({'one': rev1}, b.tags.get_tag_dict())

    def make_write_locked_branch_with_one_tag(self):
        b, revids = self.make_branch_with_revision_tuple('b', 3)
        b.tags.set_tag('one', revids[0])
        self.addCleanup(b.lock_write().unlock)
        b.tags.get_tag_dict()
        return (b, revids)

    def test_set_tag_invalides_cache(self):
        b, revids = self.make_write_locked_branch_with_one_tag()
        b.tags.set_tag('one', revids[1])
        self.assertEqual({'one': revids[1]}, b.tags.get_tag_dict())

    def test_delete_tag_invalides_cache(self):
        b, revids = self.make_write_locked_branch_with_one_tag()
        b.tags.delete_tag('one')
        self.assertEqual({}, b.tags.get_tag_dict())

    def test_merge_to_invalides_cache(self):
        b1, revids = self.make_write_locked_branch_with_one_tag()
        b2 = b1.controldir.sprout('b2').open_branch()
        b2.tags.set_tag('two', revids[1])
        b2.tags.merge_to(b1.tags)
        self.assertEqual({'one': revids[0], 'two': revids[1]}, b1.tags.get_tag_dict())

    def test_rename_revisions_invalides_cache(self):
        b, revids = self.make_write_locked_branch_with_one_tag()
        b.tags.rename_revisions({revids[0]: revids[1]})
        self.assertEqual({'one': revids[1]}, b.tags.get_tag_dict())