import errno
import os
import pwd
import shutil
import stat
import tempfile
def AtomicWrite(filename, contents, mode=438):
    """Create a file 'filename' with 'contents' atomically.

  As in Write, 'mode' is modified by the umask.  This creates and moves
  a temporary file, and errors doing the above will be propagated normally,
  though it will try to clean up the temporary file in that case.

  This is very similar to the prodlib function with the same name.

  Args:
    filename: str; the name of the file
    contents: str; the data to write to the file
    mode: int; permissions with which to create the file (default is 0666 octal)
  """
    fd, tmp_filename = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        os.write(fd, contents)
    finally:
        os.close(fd)
    try:
        os.chmod(tmp_filename, mode)
        os.rename(tmp_filename, filename)
    except OSError as exc:
        try:
            os.remove(tmp_filename)
        except OSError as e:
            exc = OSError('%s. Additional errors cleaning up: %s' % (exc, e))
        raise exc