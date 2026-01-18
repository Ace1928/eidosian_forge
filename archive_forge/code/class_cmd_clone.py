import argparse
import optparse
import os
import signal
import sys
from getopt import getopt
from typing import ClassVar, Dict, Optional, Type
from dulwich import porcelain
from .client import GitProtocolError, get_transport_and_path
from .errors import ApplyDeltaError
from .index import Index
from .objectspec import parse_commit
from .pack import Pack, sha_to_hex
from .repo import Repo
class cmd_clone(Command):

    def run(self, args):
        parser = optparse.OptionParser()
        parser.add_option('--bare', dest='bare', help='Whether to create a bare repository.', action='store_true')
        parser.add_option('--depth', dest='depth', type=int, help='Depth at which to fetch')
        parser.add_option('-b', '--branch', dest='branch', type=str, help='Check out branch instead of branch pointed to by remote HEAD')
        options, args = parser.parse_args(args)
        if args == []:
            print('usage: dulwich clone host:path [PATH]')
            sys.exit(1)
        source = args.pop(0)
        if len(args) > 0:
            target = args.pop(0)
        else:
            target = None
        try:
            porcelain.clone(source, target, bare=options.bare, depth=options.depth, branch=options.branch)
        except GitProtocolError as e:
            print('%s' % e)