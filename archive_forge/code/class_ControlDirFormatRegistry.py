from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlDirFormatRegistry(registry.Registry[str, ControlDirFormat]):
    """Registry of user-selectable ControlDir subformats.

    Differs from ControlDirFormat._formats in that it provides sub-formats,
    e.g. BzrDirMeta1 with weave repository.  Also, it's more user-oriented.
    """

    def __init__(self):
        """Create a ControlDirFormatRegistry."""
        self._registration_order = list()
        super().__init__()

    def register(self, key, factory, help, native=True, deprecated=False, hidden=False, experimental=False):
        """Register a ControlDirFormat factory.

        The factory must be a callable that takes one parameter: the key.
        It must produce an instance of the ControlDirFormat when called.

        This function mainly exists to prevent the info object from being
        supplied directly.
        """
        registry.Registry.register(self, key, factory, help, ControlDirFormatInfo(native, deprecated, hidden, experimental))
        self._registration_order.append(key)

    def register_alias(self, key, target, hidden=False):
        """Register a format alias.

        Args:
          key: Alias name
          target: Target format
          hidden: Whether the alias is hidden
        """
        info = self.get_info(target)
        registry.Registry.register_alias(self, key, target, ControlDirFormatInfo(native=info.native, deprecated=info.deprecated, hidden=hidden, experimental=info.experimental))

    def register_lazy(self, key, module_name, member_name, help, native=True, deprecated=False, hidden=False, experimental=False):
        registry.Registry.register_lazy(self, key, module_name, member_name, help, ControlDirFormatInfo(native, deprecated, hidden, experimental))
        self._registration_order.append(key)

    def set_default(self, key):
        """Set the 'default' key to be a clone of the supplied key.

        This method must be called once and only once.
        """
        self.register_alias('default', key)

    def set_default_repository(self, key):
        """Set the FormatRegistry default and Repository default.

        This is a transitional method while Repository.set_default_format
        is deprecated.
        """
        if 'default' in self:
            self.remove('default')
        self.set_default(key)
        format = self.get('default')()

    def make_controldir(self, key):
        return self.get(key)()

    def help_topic(self, topic):
        output = ''
        default_realkey = None
        default_help = self.get_help('default')
        help_pairs = []
        for key in self._registration_order:
            if key == 'default':
                continue
            help = self.get_help(key)
            if help == default_help:
                default_realkey = key
            else:
                help_pairs.append((key, help))

        def wrapped(key, help, info):
            if info.native:
                help = '(native) ' + help
            return ':{}:\n{}\n\n'.format(key, textwrap.fill(help, initial_indent='    ', subsequent_indent='    ', break_long_words=False))
        if default_realkey is not None:
            output += wrapped(default_realkey, '(default) %s' % default_help, self.get_info('default'))
        deprecated_pairs = []
        experimental_pairs = []
        for key, help in help_pairs:
            info = self.get_info(key)
            if info.hidden:
                continue
            elif info.deprecated:
                deprecated_pairs.append((key, help))
            elif info.experimental:
                experimental_pairs.append((key, help))
            else:
                output += wrapped(key, help, info)
        output += '\nSee :doc:`formats-help` for more about storage formats.'
        other_output = ''
        if len(experimental_pairs) > 0:
            other_output += 'Experimental formats are shown below.\n\n'
            for key, help in experimental_pairs:
                info = self.get_info(key)
                other_output += wrapped(key, help, info)
        else:
            other_output += 'No experimental formats are available.\n\n'
        if len(deprecated_pairs) > 0:
            other_output += '\nDeprecated formats are shown below.\n\n'
            for key, help in deprecated_pairs:
                info = self.get_info(key)
                other_output += wrapped(key, help, info)
        else:
            other_output += '\nNo deprecated formats are available.\n\n'
        other_output += '\nSee :doc:`formats-help` for more about storage formats.'
        if topic == 'other-formats':
            return other_output
        else:
            return output