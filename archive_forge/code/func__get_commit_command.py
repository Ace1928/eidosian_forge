import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _get_commit_command(self, git_ref, mark, revobj, file_cmds):
    committer = revobj.committer
    name, email = self._get_name_email(committer)
    committer_info = (name, email, revobj.timestamp, revobj.timezone)
    if self._multi_author_api_available:
        more_authors = revobj.get_apparent_authors()
        author = more_authors.pop(0)
    else:
        more_authors = []
        author = revobj.get_apparent_author()
    if not self.plain_format and more_authors:
        name, email = self._get_name_email(author)
        author_info = (name, email, revobj.timestamp, revobj.timezone)
        more_author_info = []
        for a in more_authors:
            name, email = self._get_name_email(a)
            more_author_info.append((name, email, revobj.timestamp, revobj.timezone))
    elif author != committer:
        name, email = self._get_name_email(author)
        author_info = (name, email, revobj.timestamp, revobj.timezone)
        more_author_info = None
    else:
        author_info = None
        more_author_info = None
    non_ghost_parents = []
    for p in revobj.parent_ids:
        if p in self.excluded_revisions:
            continue
        try:
            parent_mark = self.revid_to_mark[p]
            non_ghost_parents.append(b':%s' % parent_mark)
        except KeyError:
            continue
    if non_ghost_parents:
        from_ = non_ghost_parents[0]
        merges = non_ghost_parents[1:]
    else:
        from_ = None
        merges = None
    if self.plain_format:
        properties = None
    else:
        properties = revobj.properties
        for prop in self.properties_to_exclude:
            try:
                del properties[prop]
            except KeyError:
                pass
    return commands.CommitCommand(git_ref, mark, author_info, committer_info, revobj.message.encode('utf-8'), from_, merges, file_cmds, more_authors=more_author_info, properties=properties)