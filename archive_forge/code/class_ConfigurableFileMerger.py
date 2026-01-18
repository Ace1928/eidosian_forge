import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class ConfigurableFileMerger(PerFileMerger):
    """Merge individual files when configured via a .conf file.

    This is a base class for concrete custom file merging logic. Concrete
    classes should implement ``merge_text``.

    See ``breezy.plugins.news_merge.news_merge`` for an example concrete class.

    :ivar affected_files: The configured file paths to merge.

    :cvar name_prefix: The prefix to use when looking up configuration
        details. <name_prefix>_merge_files describes the files targeted by the
        hook for example.

    :cvar default_files: The default file paths to merge when no configuration
        is present.
    """
    name_prefix: str
    default_files = None

    def __init__(self, merger):
        super().__init__(merger)
        self.affected_files = None
        self.default_files = self.__class__.default_files or []
        self.name_prefix = self.__class__.name_prefix
        if self.name_prefix is None:
            raise ValueError('name_prefix must be set.')

    def file_matches(self, params):
        """Check whether the file should call the merge hook.

        <name_prefix>_merge_files configuration variable is a list of files
        that should use the hook.
        """
        affected_files = self.affected_files
        if affected_files is None:
            config = self.merger.this_branch.get_config()
            config_key = self.name_prefix + '_merge_files'
            affected_files = config.get_user_option_as_list(config_key)
            if affected_files is None:
                affected_files = self.default_files
            self.affected_files = affected_files
        if affected_files:
            filepath = params.this_path
            if filepath in affected_files:
                return True
        return False

    def merge_matching(self, params):
        return self.merge_text(params)

    def merge_text(self, params):
        """Merge the byte contents of a single file.

        This is called after checking that the merge should be performed in
        merge_contents, and it should behave as per
        ``breezy.merge.AbstractPerFileMerger.merge_contents``.
        """
        raise NotImplementedError(self.merge_text)