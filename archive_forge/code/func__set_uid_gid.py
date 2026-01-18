import os
from warnings import warn
import sys
from distutils.errors import DistutilsExecError
from distutils.spawn import spawn
from distutils.dir_util import mkpath
from distutils import log
def _set_uid_gid(tarinfo):
    if gid is not None:
        tarinfo.gid = gid
        tarinfo.gname = group
    if uid is not None:
        tarinfo.uid = uid
        tarinfo.uname = owner
    return tarinfo