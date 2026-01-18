import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
class GitShaMap:
    """Git<->Bzr revision id mapping database."""

    def lookup_git_sha(self, sha):
        """Lookup a Git sha in the database.
        :param sha: Git object sha
        :return: list with (type, type_data) tuples with type_data:
            commit: revid, tree_sha, verifiers
            blob: fileid, revid
            tree: fileid, revid
        """
        raise NotImplementedError(self.lookup_git_sha)

    def lookup_blob_id(self, file_id, revision):
        """Retrieve a Git blob SHA by file id.

        :param file_id: File id of the file/symlink
        :param revision: revision in which the file was last changed.
        """
        raise NotImplementedError(self.lookup_blob_id)

    def lookup_tree_id(self, file_id, revision):
        """Retrieve a Git tree SHA by file id.
        """
        raise NotImplementedError(self.lookup_tree_id)

    def lookup_commit(self, revid):
        """Retrieve a Git commit SHA by Bazaar revision id.
        """
        raise NotImplementedError(self.lookup_commit)

    def revids(self):
        """List the revision ids known."""
        raise NotImplementedError(self.revids)

    def missing_revisions(self, revids):
        """Return set of all the revisions that are not present."""
        present_revids = set(self.revids())
        if not isinstance(revids, set):
            revids = set(revids)
        return revids - present_revids

    def sha1s(self):
        """List the SHA1s."""
        raise NotImplementedError(self.sha1s)

    def start_write_group(self):
        """Start writing changes."""

    def commit_write_group(self):
        """Commit any pending changes."""

    def abort_write_group(self):
        """Abort any pending changes."""