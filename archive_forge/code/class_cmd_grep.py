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
class cmd_grep(Command):
    """Print lines matching PATTERN for specified files and revisions.

    This command searches the specified files and revisions for a given
    pattern.  The pattern is specified as a Python regular expressions[1].

    If the file name is not specified, the revisions starting with the
    current directory are searched recursively. If the revision number is
    not specified, the working copy is searched. To search the last committed
    revision, use the '-r -1' or '-r last:1' option.

    Unversioned files are not searched unless explicitly specified on the
    command line. Unversioned directores are not searched.

    When searching a pattern, the output is shown in the 'filepath:string'
    format. If a revision is explicitly searched, the output is shown as
    'filepath~N:string', where N is the revision number.

    --include and --exclude options can be used to search only (or exclude
    from search) files with base name matches the specified Unix style GLOB
    pattern.  The GLOB pattern an use *, ?, and [...] as wildcards, and \\
    to quote wildcard or backslash character literally. Note that the glob
    pattern is not a regular expression.

    [1] http://docs.python.org/library/re.html#regular-expression-syntax
    """
    encoding_type = 'replace'
    takes_args = ['pattern', 'path*']
    takes_options = ['verbose', 'revision', Option('color', type=str, argname='when', help='Show match in color. WHEN is never, always or auto.'), Option('diff', short_name='p', help='Grep for pattern in changeset for each revision.'), ListOption('exclude', type=str, argname='glob', short_name='X', help='Skip files whose base name matches GLOB.'), ListOption('include', type=str, argname='glob', short_name='I', help='Search only files whose base name matches GLOB.'), Option('files-with-matches', short_name='l', help='Print only the name of each input file in which PATTERN is found.'), Option('files-without-match', short_name='L', help='Print only the name of each input file in which PATTERN is not found.'), Option('fixed-string', short_name='F', help='Interpret PATTERN is a single fixed string (not regex).'), Option('from-root', help='Search for pattern starting from the root of the branch. (implies --recursive)'), Option('ignore-case', short_name='i', help='Ignore case distinctions while matching.'), Option('levels', help='Number of levels to display - 0 for all, 1 for collapsed (1 is default).', argname='N', type=_parse_levels), Option('line-number', short_name='n', help='Show 1-based line number.'), Option('no-recursive', help="Don't recurse into subdirectories. (default is --recursive)"), Option('null', short_name='Z', help='Write an ASCII NUL (\\0) separator between output lines rather than a newline.')]

    @display_command
    def run(self, verbose=False, ignore_case=False, no_recursive=False, from_root=False, null=False, levels=None, line_number=False, path_list=None, revision=None, pattern=None, include=None, exclude=None, fixed_string=False, files_with_matches=False, files_without_match=False, color=None, diff=False):
        import re
        from breezy import _termcolor
        from . import grep
        if path_list is None:
            path_list = ['.']
        elif from_root:
            raise errors.CommandError('cannot specify both --from-root and PATH.')
        if files_with_matches and files_without_match:
            raise errors.CommandError('cannot specify both -l/--files-with-matches and -L/--files-without-matches.')
        global_config = _mod_config.GlobalConfig()
        if color is None:
            color = global_config.get_user_option('grep_color')
        if color is None:
            color = 'never'
        if color not in ['always', 'never', 'auto']:
            raise errors.CommandError('Valid values for --color are "always", "never" or "auto".')
        if levels is None:
            levels = 1
        print_revno = False
        if revision is not None or levels == 0:
            print_revno = True
        eol_marker = '\n'
        if null:
            eol_marker = '\x00'
        if not ignore_case and grep.is_fixed_string(pattern):
            fixed_string = True
        elif ignore_case and fixed_string:
            fixed_string = False
            pattern = re.escape(pattern)
        patternc = None
        re_flags = re.MULTILINE
        if ignore_case:
            re_flags |= re.IGNORECASE
        if not fixed_string:
            patternc = grep.compile_pattern(pattern.encode(grep._user_encoding), re_flags)
        if color == 'always':
            show_color = True
        elif color == 'never':
            show_color = False
        elif color == 'auto':
            show_color = _termcolor.allow_color()
        opts = grep.GrepOptions()
        opts.verbose = verbose
        opts.ignore_case = ignore_case
        opts.no_recursive = no_recursive
        opts.from_root = from_root
        opts.null = null
        opts.levels = levels
        opts.line_number = line_number
        opts.path_list = path_list
        opts.revision = revision
        opts.pattern = pattern
        opts.include = include
        opts.exclude = exclude
        opts.fixed_string = fixed_string
        opts.files_with_matches = files_with_matches
        opts.files_without_match = files_without_match
        opts.color = color
        opts.diff = False
        opts.eol_marker = eol_marker
        opts.print_revno = print_revno
        opts.patternc = patternc
        opts.recursive = not no_recursive
        opts.fixed_string = fixed_string
        opts.outf = self.outf
        opts.show_color = show_color
        if diff:
            grep.grep_diff(opts)
        elif revision is None:
            grep.workingtree_grep(opts)
        else:
            grep.versioned_grep(opts)