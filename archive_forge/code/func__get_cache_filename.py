import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def _get_cache_filename(self, bucket: Bucket) -> str:
    return os.path.join(self.directory, self.pattern % (bucket.key,))