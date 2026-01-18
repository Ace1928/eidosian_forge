import copy
import ntpath
from collections import namedtuple
from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import (
from ..types import HostConfig
from ..utils import version_gte
from .images import Image
from .resource import Collection, Model
def _host_volume_from_bind(bind):
    drive, rest = ntpath.splitdrive(bind)
    bits = rest.split(':', 1)
    if len(bits) == 1 or bits[1] in ('ro', 'rw'):
        return drive + bits[0]
    elif bits[1].endswith(':ro') or bits[1].endswith(':rw'):
        return bits[1][:-3]
    else:
        return bits[1]