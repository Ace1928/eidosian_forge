from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlDirFormat:
    """An encapsulation of the initialization and open routines for a format.

    Formats provide three things:
     * An initialization routine,
     * a format string,
     * an open routine.

    Formats are placed in a dict by their format string for reference
    during controldir opening. These should be subclasses of ControlDirFormat
    for consistency.

    Once a format is deprecated, just deprecate the initialize and open
    methods on the format class. Do not deprecate the object, as the
    object will be created every system load.

    Attributes:
      colocated_branches: Whether this formats supports colocated branches.
      supports_workingtrees: This control directory can co-exist with a
                                 working tree.
    """
    _default_format: Optional['ControlDirFormat'] = None
    'The default format used for new control directories.'
    _probers: List[Type['Prober']] = []
    'The registered format probers, e.g. BzrProber.\n\n    This is a list of Prober-derived classes.\n    '
    colocated_branches = False
    'Whether co-located branches are supported for this control dir format.\n    '
    supports_workingtrees = True
    'Whether working trees can exist in control directories of this format.\n    '
    fixed_components = False
    'Whether components can not change format independent of the control dir.\n    '
    upgrade_recommended = False
    'Whether an upgrade from this format is recommended.'

    def get_format_description(self):
        """Return the short description for this format."""
        raise NotImplementedError(self.get_format_description)

    def get_converter(self, format=None):
        """Return the converter to use to convert controldirs needing converts.

        This returns a breezy.controldir.Converter object.

        This should return the best upgrader to step this format towards the
        current default format. In the case of plugins we can/should provide
        some means for them to extend the range of returnable converters.

        Args:
          format: Optional format to override the default format of the
                       library.
        """
        raise NotImplementedError(self.get_converter)

    def is_supported(self):
        """Is this format supported?

        Supported formats must be openable.
        Unsupported formats may not support initialization or committing or
        some other features depending on the reason for not being supported.
        """
        return True

    def is_initializable(self):
        """Whether new control directories of this format can be initialized.
        """
        return self.is_supported()

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        """Give an error or warning on old formats.

        Args:
          allow_unsupported: If true, allow opening
            formats that are strongly deprecated, and which may
            have limited functionality.

          recommend_upgrade: If true (default), warn
            the user through the ui object that they may wish
            to upgrade the object.
        """
        if not allow_unsupported and (not self.is_supported()):
            raise errors.UnsupportedFormatError(format=self)
        if recommend_upgrade and self.upgrade_recommended:
            ui.ui_factory.recommend_upgrade(self.get_format_description(), basedir)

    def same_model(self, target_format):
        return self.repository_format.rich_root_data == target_format.rich_root_data

    @classmethod
    def register_prober(klass, prober: Type['Prober']):
        """Register a prober that can look for a control dir.

        """
        klass._probers.append(prober)

    @classmethod
    def unregister_prober(klass, prober: Type['Prober']):
        """Unregister a prober.

        """
        klass._probers.remove(prober)

    def __str__(self):
        return self.get_format_description().rstrip()

    @classmethod
    def all_probers(klass) -> List[Type['Prober']]:
        return klass._probers

    @classmethod
    def known_formats(klass):
        """Return all the known formats.
        """
        result = []
        for prober_kls in klass.all_probers():
            result.extend(prober_kls.known_formats())
        return result

    @classmethod
    def find_format(klass, transport: _mod_transport.Transport, probers: Optional[List[Type['Prober']]]=None) -> 'ControlDirFormat':
        """Return the format present at transport."""
        if probers is None:
            probers = sorted(klass.all_probers(), key=lambda prober: prober.priority(transport))
        for prober_kls in probers:
            prober = prober_kls()
            try:
                return prober.probe_transport(transport)
            except errors.NotBranchError:
                pass
        raise errors.NotBranchError(path=transport.base)

    def initialize(self, url, possible_transports=None):
        """Create a control dir at this url and return an opened copy.

        While not deprecated, this method is very specific and its use will
        lead to many round trips to setup a working environment. See
        initialize_on_transport_ex for a [nearly] all-in-one method.

        Subclasses should typically override initialize_on_transport
        instead of this method.
        """
        return self.initialize_on_transport(_mod_transport.get_transport(url, possible_transports))

    def initialize_on_transport(self, transport):
        """Initialize a new controldir in the base directory of a Transport."""
        raise NotImplementedError(self.initialize_on_transport)

    def initialize_on_transport_ex(self, transport, use_existing_dir=False, create_prefix=False, force_new_repo=False, stacked_on=None, stack_on_pwd=None, repo_format_name=None, make_working_trees=None, shared_repo=False, vfs_only=False):
        """Create this format on transport.

        The directory to initialize will be created.

        Args:
          force_new_repo: Do not use a shared repository for the target,
                               even if one is available.
          create_prefix: Create any missing directories leading up to
            to_transport.
          use_existing_dir: Use an existing directory if one exists.
          stacked_on: A url to stack any created branch on, None to follow
            any target stacking policy.
          stack_on_pwd: If stack_on is relative, the location it is
            relative to.
          repo_format_name: If non-None, a repository will be
            made-or-found. Should none be found, or if force_new_repo is True
            the repo_format_name is used to select the format of repository to
            create.
          make_working_trees: Control the setting of make_working_trees
            for a new shared repository when one is made. None to use whatever
            default the format has.
          shared_repo: Control whether made repositories are shared or
            not.
          vfs_only: If True do not attempt to use a smart server

        Returns: repo, controldir, require_stacking, repository_policy. repo is
            None if none was created or found, controldir is always valid.
            require_stacking is the result of examining the stacked_on
            parameter and any stacking policy found for the target.
        """
        raise NotImplementedError(self.initialize_on_transport_ex)

    def network_name(self):
        """A simple byte string uniquely identifying this format for RPC calls.

        Bzr control formats use this disk format string to identify the format
        over the wire. Its possible that other control formats have more
        complex detection requirements, so we permit them to use any unique and
        immutable string they desire.
        """
        raise NotImplementedError(self.network_name)

    def open(self, transport: _mod_transport.Transport, _found=False) -> 'ControlDir':
        """Return an instance of this format for the dir transport points at.
        """
        raise NotImplementedError(self.open)

    @classmethod
    def _set_default_format(klass, format):
        """Set default format (for testing behavior of defaults only)"""
        klass._default_format = format

    @classmethod
    def get_default_format(klass):
        """Return the current default format."""
        return klass._default_format

    def supports_transport(self, transport):
        """Check if this format can be opened over a particular transport.
        """
        raise NotImplementedError(self.supports_transport)

    @classmethod
    def is_control_filename(klass, filename):
        """True if filename is the name of a path which is reserved for
        controldirs.

        Args:
          filename: A filename within the root transport of this
            controldir.

        This is true IF and ONLY IF the filename is part of the namespace reserved
        for bzr control dirs. Currently this is the '.bzr' directory in the root
        of the root_transport. it is expected that plugins will need to extend
        this in the future - for instance to make bzr talk with svn working
        trees.
        """
        raise NotImplementedError(cls.is_control_filename)