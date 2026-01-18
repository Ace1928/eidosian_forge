from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
class CommitBuilder:
    """Provides an interface to build up a commit.

    This allows describing a tree to be committed without needing to
    know the internals of the format of the repository.
    """
    record_root_entry = True
    updates_branch = False

    def __init__(self, repository, parents, config_stack, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
        """Initiate a CommitBuilder.

        Args:
          repository: Repository to commit to.
          parents: Revision ids of the parents of the new revision.
          timestamp: Optional timestamp recorded for commit.
          timezone: Optional timezone for timestamp.
          committer: Optional committer to set for commit.
          revprops: Optional dictionary of revision properties.
          revision_id: Optional revision id.
          lossy: Whether to discard data that can not be natively
            represented, when pushing to a foreign VCS
        """
        self._config_stack = config_stack
        self._lossy = lossy
        if committer is None:
            self._committer = self._config_stack.get('email')
        elif not isinstance(committer, str):
            self._committer = committer.decode()
        else:
            self._committer = committer
        self.parents = parents
        self.repository = repository
        self._revprops = {}
        if revprops is not None:
            self._validate_revprops(revprops)
            self._revprops.update(revprops)
        if timestamp is None:
            timestamp = time.time()
        self._timestamp = round(timestamp, 3)
        if timezone is None:
            self._timezone = osutils.local_time_offset()
        else:
            self._timezone = int(timezone)
        self._generate_revision_if_needed(revision_id)

    def any_changes(self):
        """Return True if any entries were changed.

        This includes merge-only changes. It is the core for the --unchanged
        detection in commit.

        Returns: True if any changes have occured.
        """
        raise NotImplementedError(self.any_changes)

    def _validate_unicode_text(self, text, context):
        """Verify things like commit messages don't have bogus characters."""
        if '\r' in text:
            raise ValueError('Invalid value for {}: {!r}'.format(context, text))

    def _validate_revprops(self, revprops):
        for key, value in revprops.items():
            if not isinstance(value, str):
                raise ValueError('revision property (%s) is not a valid (unicode) string: %r' % (key, value))
            self._validate_unicode_text(value, 'revision property ({})'.format(key))

    def commit(self, message):
        """Make the actual commit.

        Returns: The revision id of the recorded revision.
        """
        raise NotImplementedError(self.commit)

    def abort(self):
        """Abort the commit that is being built.
        """
        raise NotImplementedError(self.abort)

    def revision_tree(self) -> 'RevisionTree':
        """Return the tree that was just committed.

        After calling commit() this can be called to get a
        RevisionTree representing the newly committed tree. This is
        preferred to calling Repository.revision_tree() because that may
        require deserializing the inventory, while we already have a copy in
        memory.
        """
        raise NotImplementedError(self.revision_tree)

    def finish_inventory(self):
        """Tell the builder that the inventory is finished.

        Returns: The inventory id in the repository, which can be used with
            repository.get_inventory.
        """
        raise NotImplementedError(self.finish_inventory)

    def _generate_revision_if_needed(self, revision_id):
        """Create a revision id if None was supplied.

        If the repository can not support user-specified revision ids
        they should override this function and raise CannotSetRevisionId
        if _new_revision_id is not None.

        Raises:
          CannotSetRevisionId
        """
        if not self.repository._format.supports_setting_revision_ids:
            if revision_id is not None:
                raise CannotSetRevisionId()
            return
        if revision_id is None:
            self._new_revision_id = self._gen_revision_id()
            self.random_revid = True
        else:
            self._new_revision_id = revision_id
            self.random_revid = False

    def record_iter_changes(self, tree, basis_revision_id, iter_changes):
        """Record a new tree via iter_changes.

        Args:
          tree: The tree to obtain text contents from for changed objects.
          basis_revision_id: The revision id of the tree the iter_changes
            has been generated against. Currently assumed to be the same
            as self.parents[0] - if it is not, errors may occur.
          iter_changes: An iter_changes iterator with the changes to apply
            to basis_revision_id. The iterator must not include any items with
            a current kind of None - missing items must be either filtered out
            or errored-on beefore record_iter_changes sees the item.
        Returns: A generator of (relpath, fs_hash) tuples for use with
            tree._observed_sha1.
        """
        raise NotImplementedError(self.record_iter_changes)