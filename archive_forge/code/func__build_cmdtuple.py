import os
import errno
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log
def _build_cmdtuple(path, cmdtuples):
    """Helper for remove_tree()."""
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        if os.path.isdir(real_f) and (not os.path.islink(real_f)):
            _build_cmdtuple(real_f, cmdtuples)
        else:
            cmdtuples.append((os.remove, real_f))
    cmdtuples.append((os.rmdir, path))