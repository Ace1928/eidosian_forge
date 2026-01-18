import errno
import os
import pwd
import shutil
import stat
import tempfile
def MkDirs(directory, force_mode=None):
    """Makes a directory including its parent directories.

  This function is equivalent to os.makedirs() but it avoids a race
  condition that os.makedirs() has.  The race is between os.mkdir() and
  os.path.exists() which fail with errors when run in parallel.

  Args:
    directory: str; the directory to make
    force_mode: optional octal, chmod dir to get rid of umask interaction
  Raises:
    Whatever os.mkdir() raises when it fails for any reason EXCLUDING
    "dir already exists".  If a directory already exists, it does not
    raise anything.  This behaviour is different than os.makedirs()
  """
    name = os.path.normpath(directory)
    dirs = name.split(os.path.sep)
    for i in range(0, len(dirs)):
        path = os.path.sep.join(dirs[:i + 1])
        try:
            if path:
                os.mkdir(path)
                if force_mode is not None:
                    os.chmod(path, force_mode)
        except OSError as exc:
            if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
                raise