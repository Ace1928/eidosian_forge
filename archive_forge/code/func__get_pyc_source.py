import _frozen_importlib_external as _bootstrap_external
from _frozen_importlib_external import _unpack_uint16, _unpack_uint32
import _frozen_importlib as _bootstrap  # for _verbose_message
import _imp  # for check_hash_based_pycs
import _io  # for open
import marshal  # for loads
import sys  # for modules
import time  # for mktime
import _warnings  # For warn()
def _get_pyc_source(self, path):
    assert path[-1:] in ('c', 'o')
    path = path[:-1]
    try:
        toc_entry = self._files[path]
    except KeyError:
        return None
    else:
        return _get_data(self.archive, toc_entry)