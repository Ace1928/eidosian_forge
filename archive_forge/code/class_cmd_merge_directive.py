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
class cmd_merge_directive(Command):
    __doc__ = 'Generate a merge directive for auto-merge tools.\n\n    A directive requests a merge to be performed, and also provides all the\n    information necessary to do so.  This means it must either include a\n    revision bundle, or the location of a branch containing the desired\n    revision.\n\n    A submit branch (the location to merge into) must be supplied the first\n    time the command is issued.  After it has been supplied once, it will\n    be remembered as the default.\n\n    A public branch is optional if a revision bundle is supplied, but required\n    if --diff or --plain is specified.  It will be remembered as the default\n    after the first use.\n    '
    takes_args = ['submit_branch?', 'public_branch?']
    hidden = True
    _see_also = ['send']
    takes_options = ['directory', RegistryOption.from_kwargs('patch-type', 'The type of patch to include in the directive.', title='Patch type', value_switches=True, enum_switch=False, bundle='Bazaar revision bundle (default).', diff='Normal unified diff.', plain='No patch, just directive.'), Option('sign', help='GPG-sign the directive.'), 'revision', Option('mail-to', type=str, help='Instead of printing the directive, email to this address.'), Option('message', type=str, short_name='m', help='Message to use when committing this merge.')]
    encoding_type = 'exact'

    def run(self, submit_branch=None, public_branch=None, patch_type='bundle', sign=False, revision=None, mail_to=None, message=None, directory='.'):
        from .revision import NULL_REVISION
        include_patch, include_bundle = {'plain': (False, False), 'diff': (True, False), 'bundle': (True, True)}[patch_type]
        branch = Branch.open(directory)
        stored_submit_branch = branch.get_submit_branch()
        if submit_branch is None:
            submit_branch = stored_submit_branch
        elif stored_submit_branch is None:
            branch.set_submit_branch(submit_branch)
        if submit_branch is None:
            submit_branch = branch.get_parent()
        if submit_branch is None:
            raise errors.CommandError(gettext('No submit branch specified or known'))
        stored_public_branch = branch.get_public_branch()
        if public_branch is None:
            public_branch = stored_public_branch
        elif stored_public_branch is None:
            branch.set_public_branch(public_branch)
        if not include_bundle and public_branch is None:
            raise errors.CommandError(gettext('No public branch specified or known'))
        base_revision_id = None
        if revision is not None:
            if len(revision) > 2:
                raise errors.CommandError(gettext('brz merge-directive takes at most two one revision identifiers'))
            revision_id = revision[-1].as_revision_id(branch)
            if len(revision) == 2:
                base_revision_id = revision[0].as_revision_id(branch)
        else:
            revision_id = branch.last_revision()
        if revision_id == NULL_REVISION:
            raise errors.CommandError(gettext('No revisions to bundle.'))
        directive = merge_directive.MergeDirective2.from_objects(branch.repository, revision_id, time.time(), osutils.local_time_offset(), submit_branch, public_branch=public_branch, include_patch=include_patch, include_bundle=include_bundle, message=message, base_revision_id=base_revision_id)
        if mail_to is None:
            if sign:
                self.outf.write(directive.to_signed(branch))
            else:
                self.outf.writelines(directive.to_lines())
        else:
            message = directive.to_email(mail_to, branch, sign)
            s = SMTPConnection(branch.get_config_stack())
            s.send_email(message)