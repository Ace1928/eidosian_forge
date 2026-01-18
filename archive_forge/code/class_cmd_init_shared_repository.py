import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_init_shared_repository(Command):
    __doc__ = 'Create a shared repository for branches to share storage space.\n\n    New branches created under the repository directory will store their\n    revisions in the repository, not in the branch directory.  For branches\n    with shared history, this reduces the amount of storage needed and\n    speeds up the creation of new branches.\n\n    If the --no-trees option is given then the branches in the repository\n    will not have working trees by default.  They will still exist as\n    directories on disk, but they will not have separate copies of the\n    files at a certain revision.  This can be useful for repositories that\n    store branches which are interacted with through checkouts or remote\n    branches, such as on a server.\n\n    :Examples:\n        Create a shared repository holding just branches::\n\n            brz init-shared-repo --no-trees repo\n            brz init repo/trunk\n\n        Make a lightweight checkout elsewhere::\n\n            brz checkout --lightweight repo/trunk trunk-checkout\n            cd trunk-checkout\n            (add files here)\n    '
    _see_also = ['init', 'branch', 'checkout', 'repositories']
    takes_args = ['location']
    takes_options = [RegistryOption('format', help='Specify a format for this repository. See "brz help formats" for details.', lazy_registry=('breezy.controldir', 'format_registry'), converter=lambda name: controldir.format_registry.make_controldir(name), value_switches=True, title='Repository format'), Option('no-trees', help='Branches in the repository will default to not having a working tree.')]
    aliases = ['init-shared-repo', 'init-repo']

    def run(self, location, format=None, no_trees=False):
        if format is None:
            format = controldir.format_registry.make_controldir('default')
        if location is None:
            location = '.'
        to_transport = transport.get_transport(location, purpose='write')
        if format.fixed_components:
            repo_format_name = None
        else:
            repo_format_name = format.repository_format.get_format_string()
        repo, newdir, require_stacking, repository_policy = format.initialize_on_transport_ex(to_transport, create_prefix=True, make_working_trees=not no_trees, shared_repo=True, force_new_repo=True, use_existing_dir=True, repo_format_name=repo_format_name)
        if not is_quiet():
            from .info import show_bzrdir_info
            show_bzrdir_info(newdir, verbose=0, outfile=self.outf)