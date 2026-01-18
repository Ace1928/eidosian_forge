from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _get_extract_tarinfo(self, member, filter_function, path):
    """Get filtered TarInfo (or None) from member, which might be a str"""
    if isinstance(member, str):
        tarinfo = self.getmember(member)
    else:
        tarinfo = member
    unfiltered = tarinfo
    try:
        tarinfo = filter_function(tarinfo, path)
    except (OSError, FilterError) as e:
        self._handle_fatal_error(e)
    except ExtractError as e:
        self._handle_nonfatal_error(e)
    if tarinfo is None:
        self._dbg(2, 'tarfile: Excluded %r' % unfiltered.name)
        return None
    if tarinfo.islnk():
        tarinfo = copy.copy(tarinfo)
        tarinfo._link_target = os.path.join(path, tarinfo.linkname)
    return tarinfo