import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def _file_grep_list_only_wtree(file, path, opts, path_prefix=None):
    if b'\x00' in file.read(1024):
        if opts.verbose:
            trace.warning("Binary file '%s' skipped.", path)
        return
    file.seek(0)
    found = False
    if opts.fixed_string:
        pattern = opts.pattern.encode(_user_encoding, 'replace')
        for line in file:
            if pattern in line:
                found = True
                break
    else:
        for line in file:
            if opts.patternc.search(line):
                found = True
                break
    if opts.files_with_matches and found or (opts.files_without_match and (not found)):
        if path_prefix and path_prefix != '.':
            path = osutils.pathjoin(path_prefix, path)
        opts.outputter.get_writer(path, None, None)()