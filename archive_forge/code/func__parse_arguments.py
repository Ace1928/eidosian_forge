import argparse
import bdb
import locale
import multiprocessing
import os
import pdb
import sys
import traceback
from os import path
from typing import Any, List, Optional, TextIO
from docutils.utils import SystemMessage
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import Tee, format_exception_cut_frames, save_traceback
from sphinx.util.console import color_terminal, nocolor, red, terminal_safe  # type: ignore
from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.util.osutil import abspath, ensuredir
def _parse_arguments(argv: List[str]=sys.argv[1:]) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.noconfig:
        args.confdir = None
    elif not args.confdir:
        args.confdir = args.sourcedir
    if not args.doctreedir:
        args.doctreedir = os.path.join(args.outputdir, '.doctrees')
    if args.force_all and args.filenames:
        parser.error(__('cannot combine -a option and filenames'))
    if args.color == 'no' or (args.color == 'auto' and (not color_terminal())):
        nocolor()
    status: Optional[TextIO] = sys.stdout
    warning: Optional[TextIO] = sys.stderr
    error = sys.stderr
    if args.quiet:
        status = None
    if args.really_quiet:
        status = warning = None
    if warning and args.warnfile:
        try:
            warnfile = abspath(args.warnfile)
            ensuredir(path.dirname(warnfile))
            warnfp = open(args.warnfile, 'w', encoding='utf-8')
        except Exception as exc:
            parser.error(__('cannot open warning file %r: %s') % (args.warnfile, exc))
        warning = Tee(warning, warnfp)
        error = warning
    args.status = status
    args.warning = warning
    args.error = error
    confoverrides = {}
    for val in args.define:
        try:
            key, val = val.split('=', 1)
        except ValueError:
            parser.error(__('-D option argument must be in the form name=value'))
        confoverrides[key] = val
    for val in args.htmldefine:
        try:
            key, val = val.split('=')
        except ValueError:
            parser.error(__('-A option argument must be in the form name=value'))
        try:
            val = int(val)
        except ValueError:
            pass
        confoverrides['html_context.%s' % key] = val
    if args.nitpicky:
        confoverrides['nitpicky'] = True
    args.confoverrides = confoverrides
    return args