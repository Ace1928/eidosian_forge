from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
class TestPackRepositoryStacking(TestCaseWithTransport):
    """Tests for stacking pack repositories"""

    def setUp(self):
        if not self.format_supports_external_lookups:
            raise TestNotApplicable("%r doesn't support stacking" % (self.format_name,))
        super().setUp()

    def get_format(self):
        return controldir.format_registry.make_controldir(self.format_name)

    def test_stack_checks_rich_root_compatibility(self):
        repo = self.make_repository('repo', format=self.get_format())
        if repo.supports_rich_root():
            if getattr(repo._format, 'supports_tree_reference', False):
                matching_format_name = '2a'
            elif repo._format.supports_chks:
                matching_format_name = '2a'
            else:
                matching_format_name = 'rich-root-pack'
            mismatching_format_name = 'pack-0.92'
        else:
            if repo._format.supports_chks:
                raise AssertionError('no non-rich-root CHK formats known')
            else:
                matching_format_name = 'pack-0.92'
            mismatching_format_name = 'pack-0.92-subtree'
        base = self.make_repository('base', format=matching_format_name)
        repo.add_fallback_repository(base)
        bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
        e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
        self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent rich-root support')

    def test_stack_checks_serializers_compatibility(self):
        repo = self.make_repository('repo', format=self.get_format())
        if getattr(repo._format, 'supports_tree_reference', False):
            matching_format_name = '2a'
            mismatching_format_name = 'rich-root-pack'
        elif repo.supports_rich_root():
            if repo._format.supports_chks:
                matching_format_name = '2a'
            else:
                matching_format_name = 'rich-root-pack'
            mismatching_format_name = 'pack-0.92-subtree'
        else:
            raise TestNotApplicable('No formats use non-v5 serializer without having rich-root also set')
        base = self.make_repository('base', format=matching_format_name)
        repo.add_fallback_repository(base)
        bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
        e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
        self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent serializers')

    def test_adding_pack_does_not_record_pack_names_from_other_repositories(self):
        base = self.make_branch_and_tree('base', format=self.get_format())
        base.commit('foo')
        referencing = self.make_branch_and_tree('repo', format=self.get_format())
        referencing.branch.repository.add_fallback_repository(base.branch.repository)
        local_tree = referencing.branch.create_checkout('local')
        local_tree.commit('bar')
        new_instance = referencing.controldir.open_repository()
        new_instance.lock_read()
        self.addCleanup(new_instance.unlock)
        new_instance._pack_collection.ensure_loaded()
        self.assertEqual(1, len(new_instance._pack_collection.all_packs()))

    def test_autopack_only_considers_main_repo_packs(self):
        format = self.get_format()
        base = self.make_branch_and_tree('base', format=format)
        base.commit('foo')
        tree = self.make_branch_and_tree('repo', format=format)
        tree.branch.repository.add_fallback_repository(base.branch.repository)
        trans = tree.branch.repository.controldir.get_repository_transport(None)
        local_tree = tree.branch.create_checkout('local')
        for x in range(9):
            local_tree.commit('commit %s' % x)
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(9, len(list(index.iter_all_entries())))
        local_tree.commit('commit triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(1, len(list(index.iter_all_entries())))
        tree = tree.controldir.open_workingtree()
        check_result = tree.branch.repository.check([tree.branch.last_revision()])
        nb_files = 5
        if tree.branch.repository._format.supports_chks:
            nb_files += 1
        obsolete_files = list(trans.list_dir('obsolete_packs'))
        self.assertFalse('foo' in obsolete_files)
        self.assertFalse('bar' in obsolete_files)
        self.assertEqual(10 * nb_files, len(obsolete_files))
        large_pack_name = list(index.iter_all_entries())[0][1][0]
        local_tree.commit('commit not triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(2, len(list(index.iter_all_entries())))
        pack_names = [node[1][0] for node in index.iter_all_entries()]
        self.assertTrue(large_pack_name in pack_names)