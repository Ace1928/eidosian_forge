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
class cmd_tag(Command):

    def run(self, args):
        parser = optparse.OptionParser()
        parser.add_option('-a', '--annotated', help='Create an annotated tag.', action='store_true')
        parser.add_option('-s', '--sign', help='Sign the annotated tag.', action='store_true')
        options, args = parser.parse_args(args)
        porcelain.tag_create('.', args[0], annotated=options.annotated, sign=options.sign)