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
class cmd_stash(SuperCommand):
    subcommands: ClassVar[Dict[str, Type[Command]]] = {'list': cmd_stash_list, 'pop': cmd_stash_pop, 'push': cmd_stash_push}