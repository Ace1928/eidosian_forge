from collections import defaultdict
import binascii
from io import BytesIO, UnsupportedOperation
from collections import (
import difflib
import struct
from itertools import chain
import os
import sys
from hashlib import sha1
from os import (
from struct import unpack_from
import zlib
from dulwich.errors import (  # noqa: E402
from dulwich.file import GitFile  # noqa: E402
from dulwich.lru_cache import (  # noqa: E402
from dulwich.objects import (  # noqa: E402
class PackTupleIterable(object):

    def __init__(self, pack):
        self.pack = pack

    def __len__(self):
        return len(self.pack)

    def __iter__(self):
        return ((o, None) for o in self.pack.iterobjects())