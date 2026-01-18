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
class cmd_missing(Command):
    __doc__ = 'Show unmerged/unpulled revisions between two branches.\n\n    OTHER_BRANCH may be local or remote.\n\n    To filter on a range of revisions, you can use the command -r begin..end\n    -r revision requests a specific revision, -r ..end or -r begin.. are\n    also valid.\n\n    :Exit values:\n        1 - some missing revisions\n        0 - no missing revisions\n\n    :Examples:\n\n        Determine the missing revisions between this and the branch at the\n        remembered pull location::\n\n            brz missing\n\n        Determine the missing revisions between this and another branch::\n\n            brz missing http://server/branch\n\n        Determine the missing revisions up to a specific revision on the other\n        branch::\n\n            brz missing -r ..-10\n\n        Determine the missing revisions up to a specific revision on this\n        branch::\n\n            brz missing --my-revision ..-10\n    '
    _see_also = ['merge', 'pull']
    takes_args = ['other_branch?']
    takes_options = ['directory', Option('reverse', 'Reverse the order of revisions.'), Option('mine-only', 'Display changes in the local branch only.'), Option('this', 'Same as --mine-only.'), Option('theirs-only', 'Display changes in the remote branch only.'), Option('other', 'Same as --theirs-only.'), 'log-format', 'show-ids', 'verbose', custom_help('revision', help='Filter on other branch revisions (inclusive). See "help revisionspec" for details.'), Option('my-revision', type=_parse_revision_str, help='Filter on local branch revisions (inclusive). See "help revisionspec" for details.'), Option('include-merged', 'Show all revisions in addition to the mainline ones.'), Option('include-merges', hidden=True, help='Historical alias for --include-merged.')]
    encoding_type = 'replace'

    @display_command
    def run(self, other_branch=None, reverse=False, mine_only=False, theirs_only=False, log_format=None, long=False, short=False, line=False, show_ids=False, verbose=False, this=False, other=False, include_merged=None, revision=None, my_revision=None, directory='.'):
        from breezy.missing import find_unmerged, iter_log_revisions

        def message(s):
            if not is_quiet():
                self.outf.write(s)
        if include_merged is None:
            include_merged = False
        if this:
            mine_only = this
        if other:
            theirs_only = other
        restrict = 'all'
        if mine_only:
            restrict = 'local'
        elif theirs_only:
            restrict = 'remote'
        local_branch = Branch.open_containing(directory)[0]
        self.enter_context(local_branch.lock_read())
        parent = local_branch.get_parent()
        if other_branch is None:
            other_branch = parent
            if other_branch is None:
                raise errors.CommandError(gettext('No peer location known or specified.'))
            display_url = urlutils.unescape_for_display(parent, self.outf.encoding)
            message(gettext('Using saved parent location: {0}\n').format(display_url))
        remote_branch = Branch.open(other_branch)
        if remote_branch.base == local_branch.base:
            remote_branch = local_branch
        else:
            self.enter_context(remote_branch.lock_read())
        local_revid_range = _revision_range_to_revid_range(_get_revision_range(my_revision, local_branch, self.name()))
        remote_revid_range = _revision_range_to_revid_range(_get_revision_range(revision, remote_branch, self.name()))
        local_extra, remote_extra = find_unmerged(local_branch, remote_branch, restrict, backward=not reverse, include_merged=include_merged, local_revid_range=local_revid_range, remote_revid_range=remote_revid_range)
        if log_format is None:
            registry = log.log_formatter_registry
            log_format = registry.get_default(local_branch)
        lf = log_format(to_file=self.outf, show_ids=show_ids, show_timezone='original')
        status_code = 0
        if local_extra and (not theirs_only):
            message(ngettext('You have %d extra revision:\n', 'You have %d extra revisions:\n', len(local_extra)) % len(local_extra))
            rev_tag_dict = {}
            if local_branch.supports_tags():
                rev_tag_dict = local_branch.tags.get_reverse_tag_dict()
            for revision in iter_log_revisions(local_extra, local_branch.repository, verbose, rev_tag_dict):
                lf.log_revision(revision)
            printed_local = True
            status_code = 1
        else:
            printed_local = False
        if remote_extra and (not mine_only):
            if printed_local is True:
                message('\n\n\n')
            message(ngettext('You are missing %d revision:\n', 'You are missing %d revisions:\n', len(remote_extra)) % len(remote_extra))
            if remote_branch.supports_tags():
                rev_tag_dict = remote_branch.tags.get_reverse_tag_dict()
            for revision in iter_log_revisions(remote_extra, remote_branch.repository, verbose, rev_tag_dict):
                lf.log_revision(revision)
            status_code = 1
        if mine_only and (not local_extra):
            message(gettext('This branch has no new revisions.\n'))
        elif theirs_only and (not remote_extra):
            message(gettext('Other branch has no new revisions.\n'))
        elif not (mine_only or theirs_only or local_extra or remote_extra):
            message(gettext('Branches are up to date.\n'))
        self.cleanup_now()
        if not status_code and parent is None and (other_branch is not None):
            self.enter_context(local_branch.lock_write())
            if local_branch.get_parent() is None:
                local_branch.set_parent(remote_branch.base)
        return status_code