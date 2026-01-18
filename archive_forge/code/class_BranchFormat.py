from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class BranchFormat(ControlComponentFormat):
    """An encapsulation of the initialization and open routines for a format.

    Formats provide three things:
     * An initialization routine,
     * a format description
     * an open routine.

    Formats are placed in an dict by their format string for reference
    during branch opening. It's not required that these be instances, they
    can be classes themselves with class methods - it simply depends on
    whether state is needed for a given format or not.

    Once a format is deprecated, just deprecate the initialize and open
    methods on the format class. Do not deprecate the object, as the
    object will be created every time regardless.
    """

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __ne__(self, other):
        return not self == other

    def get_reference(self, controldir, name=None):
        """Get the target reference of the branch in controldir.

        format probing must have been completed before calling
        this method - it is assumed that the format of the branch
        in controldir is correct.

        Args:
          controldir: The controldir to get the branch data from.
          name: Name of the colocated branch to fetch
        Returns: None if the branch is not a reference branch.
        """
        return None

    @classmethod
    def set_reference(self, controldir, name, to_branch):
        """Set the target reference of the branch in controldir.

        format probing must have been completed before calling
        this method - it is assumed that the format of the branch
        in controldir is correct.

        Args:
          controldir: The controldir to set the branch reference for.
          name: Name of colocated branch to set, None for default
          to_branch: branch that the checkout is to reference
        """
        raise NotImplementedError(self.set_reference)

    def get_format_description(self):
        """Return the short format description for this format."""
        raise NotImplementedError(self.get_format_description)

    def _run_post_branch_init_hooks(self, controldir, name, branch):
        hooks = Branch.hooks['post_branch_init']
        if not hooks:
            return
        params = BranchInitHookParams(self, controldir, name, branch)
        for hook in hooks:
            hook(params)

    def initialize(self, controldir, name=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in controldir.

        Args:
          name: Name of the colocated branch to create.
        """
        raise NotImplementedError(self.initialize)

    def is_supported(self):
        """Is this format supported?

        Supported formats can be initialized and opened.
        Unsupported formats may not support initialization or committing or
        some other features depending on the reason for not being supported.
        """
        return True

    def make_tags(self, branch):
        """Create a tags object for branch.

        This method is on BranchFormat, because BranchFormats are reflected
        over the wire via network_name(), whereas full Branch instances require
        multiple VFS method calls to operate at all.

        The default implementation returns a disabled-tags instance.

        Note that it is normal for branch to be a RemoteBranch when using tags
        on a RemoteBranch.
        """
        from .tag import DisabledTags
        return DisabledTags(branch)

    def network_name(self):
        """A simple byte string uniquely identifying this format for RPC calls.

        MetaDir branch formats use their disk format string to identify the
        repository over the wire. All in one formats such as bzr < 0.8, and
        foreign formats like svn/git and hg should use some marker which is
        unique and immutable.
        """
        raise NotImplementedError(self.network_name)

    def open(self, controldir, name=None, _found=False, ignore_fallbacks=False, found_repository=None, possible_transports=None):
        """Return the branch object for controldir.

        Args:
          controldir: A ControlDir that contains a branch.
          name: Name of colocated branch to open
          _found: a private parameter, do not use it. It is used to
            indicate if format probing has already be done.
          ignore_fallbacks: when set, no fallback branches will be opened
            (if there are any).  Default is to open fallbacks.
        """
        raise NotImplementedError(self.open)

    def supports_set_append_revisions_only(self):
        """True if this format supports set_append_revisions_only."""
        return False

    def supports_stacking(self):
        """True if this format records a stacked-on branch."""
        return False

    def supports_leaving_lock(self):
        """True if this format supports leaving locks in place."""
        return False

    def __str__(self):
        return self.get_format_description().rstrip()

    def supports_tags(self):
        """True if this format supports tags stored in the branch"""
        return False

    def tags_are_versioned(self):
        """Whether the tag container for this branch versions tags."""
        return False

    def supports_tags_referencing_ghosts(self):
        """True if tags can reference ghost revisions."""
        return True

    def supports_store_uncommitted(self):
        """True if uncommitted changes can be stored in this branch."""
        return True

    def stores_revno(self):
        """True if this branch format store revision numbers."""
        return True