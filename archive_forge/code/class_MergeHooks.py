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
class MergeHooks(hooks.Hooks):

    def __init__(self):
        hooks.Hooks.__init__(self, 'breezy.merge', 'Merger.hooks')
        self.add_hook('merge_file_content', 'Called with a breezy.merge.Merger object to create a per file merge object when starting a merge. Should return either None or a subclass of ``breezy.merge.AbstractPerFileMerger``. Such objects will then be called per file that needs to be merged (including when one side has deleted the file and the other has changed it). See the AbstractPerFileMerger API docs for details on how it is used by merge.', (2, 1))
        self.add_hook('pre_merge', 'Called before a merge. Receives a Merger object as the single argument.', (2, 5))
        self.add_hook('post_merge', 'Called after a merge. Receives a Merger object as the single argument. The return value is ignored.', (2, 5))