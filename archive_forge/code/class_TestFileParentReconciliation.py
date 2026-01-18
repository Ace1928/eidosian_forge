from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestFileParentReconciliation(TestCaseWithRepository):
    """Tests for how reconcile corrects errors in parents of file versions."""
    scenarios = broken_scenarios_for_all_formats()

    def make_populated_repository(self, factory):
        """Create a new repository populated by the given factory."""
        repo = self.make_repository('broken-repo')
        with repo.lock_write(), WriteGroup(repo):
            factory(repo)
            return repo

    def add_revision(self, repo, revision_id, inv, parent_ids):
        """Add a revision with a given inventory and parents to a repository.

        :param repo: a repository.
        :param revision_id: the revision ID for the new revision.
        :param inv: an inventory (such as created by
            `make_one_file_inventory`).
        :param parent_ids: the parents for the new revision.
        """
        inv.revision_id = revision_id
        inv.root.revision = revision_id
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, revision_id), [], [])
        repo.add_inventory(revision_id, inv, parent_ids)
        revision = Revision(revision_id, committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=parent_ids)
        repo.add_revision(revision_id, revision, inv)

    def make_one_file_inventory(self, repo, revision, parents, inv_revision=None, root_revision=None, file_contents=None, make_file_version=True):
        """Make an inventory containing a version of a file with ID 'a-file'.

        The file's ID will be 'a-file', and its filename will be 'a file name',
        stored at the tree root.

        :param repo: a repository to add the new file version to.
        :param revision: the revision ID of the new inventory.
        :param parents: the parents for this revision of 'a-file'.
        :param inv_revision: if not None, the revision ID to store in the
            inventory entry.  Otherwise, this defaults to revision.
        :param root_revision: if not None, the inventory's root.revision will
            be set to this.
        :param file_contents: if not None, the contents of this file version.
            Otherwise a unique default (based on revision ID) will be
            generated.
        """
        inv = Inventory(revision_id=revision)
        if root_revision is not None:
            inv.root.revision = root_revision
        file_id = b'a-file-id'
        entry = InventoryFile(file_id, 'a file name', b'TREE_ROOT')
        if inv_revision is not None:
            entry.revision = inv_revision
        else:
            entry.revision = revision
        entry.text_size = 0
        if file_contents is None:
            file_contents = b'%sline\n' % entry.revision
        entry.text_sha1 = osutils.sha_string(file_contents)
        inv.add(entry)
        if make_file_version:
            repo.texts.add_lines((file_id, revision), [(file_id, parent) for parent in parents], [file_contents])
        return inv

    def require_repo_suffers_text_parent_corruption(self, repo):
        if not repo._reconcile_fixes_text_parents:
            raise TestNotApplicable('Format does not support text parent reconciliation')

    def file_parents(self, repo, revision_id):
        key = (b'a-file-id', revision_id)
        parent_map = repo.texts.get_parent_map([key])
        return tuple((parent[-1] for parent in parent_map[key]))

    def assertFileVersionAbsent(self, repo, revision_id):
        self.assertEqual({}, repo.texts.get_parent_map([(b'a-file-id', revision_id)]))

    def assertParentsMatch(self, expected_parents_for_versions, repo, when_description):
        for expected_parents, version in expected_parents_for_versions:
            if expected_parents is None:
                self.assertFileVersionAbsent(repo, version)
            else:
                found_parents = self.file_parents(repo, version)
                self.assertEqual(expected_parents, found_parents, '%s reconcile %s has parents %s, should have %s.' % (when_description, version, found_parents, expected_parents))

    def prepare_test_repository(self):
        """Prepare a repository to test with from the test scenario.

        :return: A repository, and the scenario instance.
        """
        scenario = self.scenario_class(self)
        repo = self.make_populated_repository(scenario.populate_repository)
        self.require_repo_suffers_text_parent_corruption(repo)
        return (repo, scenario)

    def shas_for_versions_of_file(self, repo, versions):
        """Get the SHA-1 hashes of the versions of 'a-file' in the repository.

        :param repo: the repository to get the hashes from.
        :param versions: a list of versions to get hashes for.

        :returns: A dict of `{version: hash}`.
        """
        keys = [(b'a-file-id', version) for version in versions]
        return repo.texts.get_sha1s(keys)

    def test_reconcile_behaviour(self):
        """Populate a repository and reconcile it, verifying the state before
        and after.
        """
        repo, scenario = self.prepare_test_repository()
        with repo.lock_read():
            self.assertParentsMatch(scenario.populated_parents(), repo, b'before')
            vf_shas = self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile())
        result = repo.reconcile(thorough=True)
        with repo.lock_read():
            self.assertParentsMatch(scenario.corrected_parents(), repo, b'after')
            self.assertEqual(vf_shas, self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile()))
            for file_version in scenario.corrected_fulltexts():
                key = (b'a-file-id', file_version)
                self.assertEqual({key: ()}, repo.texts.get_parent_map([key]))
                self.assertIsInstance(next(repo.texts.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'), bytes)

    def test_check_behaviour(self):
        """Populate a repository and check it, and verify the output."""
        repo, scenario = self.prepare_test_repository()
        check_result = repo.check()
        check_result.report_results(verbose=True)
        log = self.get_log()
        for pattern in scenario.check_regexes(repo):
            self.assertContainsRe(log, pattern)

    def test_find_text_key_references(self):
        """Test that find_text_key_references finds erroneous references."""
        repo, scenario = self.prepare_test_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual(scenario.repository_text_key_references(), repo.find_text_key_references())

    def test__generate_text_key_index(self):
        """Test that the generated text key index has all entries."""
        repo, scenario = self.prepare_test_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual(scenario.repository_text_key_index(), repo._generate_text_key_index())