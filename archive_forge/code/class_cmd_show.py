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
class cmd_show(Command):

    def run(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('objectish', type=str, nargs='*')
        args = parser.parse_args(argv)
        porcelain.show('.', args.objectish or None)