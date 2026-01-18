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
class cmd_fetch(Command):

    def run(self, args):
        opts, args = getopt(args, '', [])
        opts = dict(opts)
        client, path = get_transport_and_path(args.pop(0))
        r = Repo('.')
        refs = client.fetch(path, r, progress=sys.stdout.write)
        print('Remote refs:')
        for item in refs.items():
            print('{} -> {}'.format(*item))