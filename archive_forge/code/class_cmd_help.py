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
class cmd_help(Command):

    def run(self, args):
        parser = optparse.OptionParser()
        parser.add_option('-a', '--all', dest='all', action='store_true', help='List all commands.')
        options, args = parser.parse_args(args)
        if options.all:
            print('Available commands:')
            for cmd in sorted(commands):
                print('  %s' % cmd)
        else:
            print("The dulwich command line tool is currently a very basic frontend for the\nDulwich python module. For full functionality, please see the API reference.\n\nFor a list of supported commands, see 'dulwich help -a'.\n")