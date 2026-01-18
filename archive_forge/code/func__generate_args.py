from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from configparser import ConfigParser
import inspect
import os
import sys
from typing import Any
from typing import cast
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Union
from typing_extensions import TypedDict
from . import __version__
from . import command
from . import util
from .util import compat
def _generate_args(self, prog: Optional[str]) -> None:

    def add_options(fn: Any, parser: Any, positional: Any, kwargs: Any) -> None:
        kwargs_opts = {'template': ('-t', '--template', dict(default='generic', type=str, help="Setup template for use with 'init'")), 'message': ('-m', '--message', dict(type=str, help="Message string to use with 'revision'")), 'sql': ('--sql', dict(action='store_true', help="Don't emit SQL to database - dump to standard output/file instead. See docs on offline mode.")), 'tag': ('--tag', dict(type=str, help="Arbitrary 'tag' name - can be used by custom env.py scripts.")), 'head': ('--head', dict(type=str, help='Specify head revision or <branchname>@head to base new revision on.')), 'splice': ('--splice', dict(action='store_true', help="Allow a non-head revision as the 'head' to splice onto")), 'depends_on': ('--depends-on', dict(action='append', help='Specify one or more revision identifiers which this revision should depend on.')), 'rev_id': ('--rev-id', dict(type=str, help='Specify a hardcoded revision id instead of generating one')), 'version_path': ('--version-path', dict(type=str, help='Specify specific path from config for version file')), 'branch_label': ('--branch-label', dict(type=str, help='Specify a branch label to apply to the new revision')), 'verbose': ('-v', '--verbose', dict(action='store_true', help='Use more verbose output')), 'resolve_dependencies': ('--resolve-dependencies', dict(action='store_true', help='Treat dependency versions as down revisions')), 'autogenerate': ('--autogenerate', dict(action='store_true', help='Populate revision script with candidate migration operations, based on comparison of database to model.')), 'rev_range': ('-r', '--rev-range', dict(action='store', help='Specify a revision range; format is [start]:[end]')), 'indicate_current': ('-i', '--indicate-current', dict(action='store_true', help='Indicate the current revision')), 'purge': ('--purge', dict(action='store_true', help='Unconditionally erase the version table before stamping')), 'package': ('--package', dict(action='store_true', help='Write empty __init__.py files to the environment and version locations'))}
        positional_help = {'directory': 'location of scripts directory', 'revision': 'revision identifier', 'revisions': "one or more revisions, or 'heads' for all heads"}
        for arg in kwargs:
            if arg in kwargs_opts:
                args = kwargs_opts[arg]
                args, kw = (args[0:-1], args[-1])
                parser.add_argument(*args, **kw)
        for arg in positional:
            if arg == 'revisions' or (fn in positional_translations and positional_translations[fn][arg] == 'revisions'):
                subparser.add_argument('revisions', nargs='+', help=positional_help.get('revisions'))
            else:
                subparser.add_argument(arg, help=positional_help.get(arg))
    parser = ArgumentParser(prog=prog)
    parser.add_argument('--version', action='version', version='%%(prog)s %s' % __version__)
    parser.add_argument('-c', '--config', type=str, default=os.environ.get('ALEMBIC_CONFIG', 'alembic.ini'), help='Alternate config file; defaults to value of ALEMBIC_CONFIG environment variable, or "alembic.ini"')
    parser.add_argument('-n', '--name', type=str, default='alembic', help='Name of section in .ini file to use for Alembic config')
    parser.add_argument('-x', action='append', help='Additional arguments consumed by custom env.py scripts, e.g. -x setting1=somesetting -x setting2=somesetting')
    parser.add_argument('--raiseerr', action='store_true', help='Raise a full stack trace on error')
    parser.add_argument('-q', '--quiet', action='store_true', help='Do not log to std output.')
    subparsers = parser.add_subparsers()
    positional_translations: Dict[Any, Any] = {command.stamp: {'revision': 'revisions'}}
    for fn in [getattr(command, n) for n in dir(command)]:
        if inspect.isfunction(fn) and fn.__name__[0] != '_' and (fn.__module__ == 'alembic.command'):
            spec = compat.inspect_getfullargspec(fn)
            if spec[3] is not None:
                positional = spec[0][1:-len(spec[3])]
                kwarg = spec[0][-len(spec[3]):]
            else:
                positional = spec[0][1:]
                kwarg = []
            if fn in positional_translations:
                positional = [positional_translations[fn].get(name, name) for name in positional]
            help_ = fn.__doc__
            if help_:
                help_text = []
                for line in help_.split('\n'):
                    if not line.strip():
                        break
                    else:
                        help_text.append(line.strip())
            else:
                help_text = []
            subparser = subparsers.add_parser(fn.__name__, help=' '.join(help_text))
            add_options(fn, subparser, positional, kwarg)
            subparser.set_defaults(cmd=(fn, positional, kwarg))
    self.parser = parser