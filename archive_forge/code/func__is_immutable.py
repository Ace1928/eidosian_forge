import os
import sys
import stat
import fnmatch
import collections
import errno
def _is_immutable(src):
    st = _stat(src)
    immutable_states = [stat.UF_IMMUTABLE, stat.SF_IMMUTABLE]
    return hasattr(st, 'st_flags') and st.st_flags in immutable_states