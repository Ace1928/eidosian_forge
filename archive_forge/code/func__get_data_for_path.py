import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def _get_data_for_path(self, path):
    if path is None:
        path = sys.path
    internal_key = tuple(path)
    if internal_key in self._internal:
        return self._internal[internal_key]
    digest, path_values = _hash_settings_for_path(path)
    filename = os.path.join(self._dir, digest)
    try:
        log.debug('reading %s', filename)
        with open(filename, 'r') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        data = _build_cacheable_data()
        data['path_values'] = path_values
        if not self._disable_caching:
            try:
                log.debug('writing to %s', filename)
                os.makedirs(self._dir, exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(data, f)
            except (IOError, OSError):
                pass
    self._internal[internal_key] = data
    return data