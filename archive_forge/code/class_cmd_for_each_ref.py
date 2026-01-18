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
class cmd_for_each_ref(Command):

    def run(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('pattern', type=str, nargs='?')
        args = parser.parse_args(args)
        for sha, object_type, ref in porcelain.for_each_ref('.', args.pattern):
            print(f'{sha.decode()} {object_type.decode()}\t{ref.decode()}')