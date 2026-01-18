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
class cmd_symbolic_ref(Command):

    def run(self, args):
        opts, args = getopt(args, '', ['ref-name', 'force'])
        if not args:
            print('Usage: dulwich symbolic-ref REF_NAME [--force]')
            sys.exit(1)
        ref_name = args.pop(0)
        porcelain.symbolic_ref('.', ref_name=ref_name, force='--force' in args)