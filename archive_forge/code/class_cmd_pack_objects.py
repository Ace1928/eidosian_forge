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
class cmd_pack_objects(Command):

    def run(self, args):
        deltify = False
        reuse_deltas = True
        opts, args = getopt(args, '', ['stdout', 'deltify', 'no-reuse-deltas'])
        opts = dict(opts)
        if len(args) < 1 and '--stdout' not in opts.keys():
            print('Usage: dulwich pack-objects basename')
            sys.exit(1)
        object_ids = [line.strip() for line in sys.stdin.readlines()]
        if '--deltify' in opts.keys():
            deltify = True
        if '--no-reuse-deltas' in opts.keys():
            reuse_deltas = False
        if '--stdout' in opts.keys():
            packf = getattr(sys.stdout, 'buffer', sys.stdout)
            idxf = None
            close = []
        else:
            basename = args[0]
            packf = open(basename + '.pack', 'wb')
            idxf = open(basename + '.idx', 'wb')
            close = [packf, idxf]
        porcelain.pack_objects('.', object_ids, packf, idxf, deltify=deltify, reuse_deltas=reuse_deltas)
        for f in close:
            f.close()