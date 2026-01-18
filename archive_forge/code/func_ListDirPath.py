import errno
import os
import pwd
import shutil
import stat
import tempfile
def ListDirPath(dir_name):
    """Like os.listdir with prepended dir_name, which is often more convenient."""
    return [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]