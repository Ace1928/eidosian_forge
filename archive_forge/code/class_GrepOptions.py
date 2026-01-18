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
class GrepOptions:
    """Container to pass around grep options.

    This class is used as a container to pass around user option and
    some other params (like outf) to processing functions. This makes
    it easier to add more options as grep evolves.
    """
    verbose = False
    ignore_case = False
    no_recursive = False
    from_root = False
    null = False
    levels = None
    line_number = False
    path_list = None
    revision = None
    pattern = None
    include = None
    exclude = None
    fixed_string = False
    files_with_matches = False
    files_without_match = False
    color = None
    diff = False
    recursive = None
    eol_marker = None
    patternc = None
    sub_patternc = None
    print_revno = None
    fixed_string = None
    outf = None
    show_color = False