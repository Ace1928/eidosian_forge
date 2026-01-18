import glob
import os
import pickle
import random
import tempfile
import time
import zlib
from hashlib import md5
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.core.files import locks
from django.core.files.move import file_move_safe
def _write_content(self, file, timeout, value):
    expiry = self.get_backend_timeout(timeout)
    file.write(pickle.dumps(expiry, self.pickle_protocol))
    file.write(zlib.compress(pickle.dumps(value, self.pickle_protocol)))