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
class RepositoryFormat(controldir.ControlComponentFormat):
    """A repository format.

    Formats provide four things:
     * An initialization routine to construct repository data on disk.
     * a optional format string which is used when the BzrDir supports
       versioned children.
     * an open routine which returns a Repository instance.
     * A network name for referring to the format in smart server RPC
       methods.

    There is one and only one Format subclass for each on-disk format. But
    there can be one Repository subclass that is used for several different
    formats. The _format attribute on a Repository instance can be used to
    determine the disk format.

    Formats are placed in a registry by their format string for reference
    during opening. These should be subclasses of RepositoryFormat for
    consistency.

    Once a format is deprecated, just deprecate the initialize and open
    methods on the format class. Do not deprecate the object, as the
    object may be created even when a repository instance hasn't been
    created.

    Common instance attributes:
    _matchingcontroldir - the controldir format that the repository format was
    originally written to work with. This can be used if manually
    constructing a bzrdir and repository, or more commonly for test suite
    parameterization.
    """
    supports_ghosts: bool
    supports_external_lookups: bool
    supports_chks: bool
    _fetch_reconcile: bool = False
    fast_deltas: bool
    pack_compresses: bool = False
    supports_tree_reference: bool
    experimental: bool = False
    supports_funky_characters: bool
    supports_leaving_lock: bool
    supports_full_versioned_files: bool
    supports_revision_signatures: bool = True
    revision_graph_can_have_wrong_parents: bool
    supports_setting_revision_ids: bool = True
    rich_root_data: bool
    supports_versioned_directories: bool
    supports_nesting_repositories: bool
    supports_unreferenced_revisions: bool
    supports_storing_branch_nick: bool = True
    supports_overriding_transport: bool = True
    supports_custom_revision_properties: bool = True
    records_per_file_revision: bool = True
    supports_multiple_authors: bool = True

    def __repr__(self):
        return '%s()' % self.__class__.__name__

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not self == other

    def get_format_description(self):
        """Return the short description for this format."""
        raise NotImplementedError(self.get_format_description)

    def initialize(self, controldir, shared=False):
        """Initialize a repository of this format in controldir.

        Args:
          controldir: The controldir to put the new repository in it.
          shared: The repository should be initialized as a sharable one.

        Returns:
          The new repository object.

        This may raise UninitializableFormat if shared repository are not
        compatible the controldir.
        """
        raise NotImplementedError(self.initialize)

    def is_supported(self):
        """Is this format supported?

        Supported formats must be initializable and openable.
        Unsupported formats may not support initialization or committing or
        some other features depending on the reason for not being supported.
        """
        return True

    def is_deprecated(self):
        """Is this format deprecated?

        Deprecated formats may trigger a user-visible warning recommending
        the user to upgrade. They are still fully supported.
        """
        return False

    def network_name(self):
        """A simple byte string uniquely identifying this format for RPC calls.

        MetaDir repository formats use their disk format string to identify the
        repository over the wire. All in one formats such as bzr < 0.8, and
        foreign formats like svn/git and hg should use some marker which is
        unique and immutable.
        """
        raise NotImplementedError(self.network_name)

    def check_conversion_target(self, target_format):
        if self.rich_root_data and (not target_format.rich_root_data):
            raise errors.BadConversionTarget('Does not support rich root data.', target_format, from_format=self)
        if self.supports_tree_reference and (not getattr(target_format, 'supports_tree_reference', False)):
            raise errors.BadConversionTarget('Does not support nested trees', target_format, from_format=self)

    def open(self, controldir, _found=False):
        """Return an instance of this format for a controldir.

        _found is a private parameter, do not use it.
        """
        raise NotImplementedError(self.open)

    def _run_post_repo_init_hooks(self, repository, controldir, shared):
        from .controldir import ControlDir, RepoInitHookParams
        hooks = ControlDir.hooks['post_repo_init']
        if not hooks:
            return
        params = RepoInitHookParams(repository, self, controldir, shared)
        for hook in hooks:
            hook(params)