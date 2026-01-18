import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
def _split_proto_line(line, allowed):
    """Split a line read from the wire.

    Args:
      line: The line read from the wire.
      allowed: An iterable of command names that should be allowed.
        Command names not listed below as possible return values will be
        ignored.  If None, any commands from the possible return values are
        allowed.
    Returns: a tuple having one of the following forms:
        ('want', obj_id)
        ('have', obj_id)
        ('done', None)
        (None, None)  (for a flush-pkt)

    Raises:
      UnexpectedCommandError: if the line cannot be parsed into one of the
        allowed return values.
    """
    if not line:
        fields = [None]
    else:
        fields = line.rstrip(b'\n').split(b' ', 1)
    command = fields[0]
    if allowed is not None and command not in allowed:
        raise UnexpectedCommandError(command)
    if len(fields) == 1 and command in (COMMAND_DONE, None):
        return (command, None)
    elif len(fields) == 2:
        if command in (COMMAND_WANT, COMMAND_HAVE, COMMAND_SHALLOW, COMMAND_UNSHALLOW):
            if not valid_hexsha(fields[1]):
                raise GitProtocolError('Invalid sha')
            return tuple(fields)
        elif command == COMMAND_DEEPEN:
            return (command, int(fields[1]))
    raise GitProtocolError('Received invalid line from client: %r' % line)