import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class FileIO(object):
    """FileIO class that exposes methods to read / write to / from files.

  The constructor takes the following arguments:
  name: [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object)
    giving the pathname of the file to be opened.
  mode: one of `r`, `w`, `a`, `r+`, `w+`, `a+`. Append `b` for bytes mode.

  Can be used as an iterator to iterate over lines in the file.

  The default buffer size used for the BufferedInputStream used for reading
  the file line by line is 1024 * 512 bytes.
  """

    def __init__(self, name, mode, encoding='utf-8'):
        self.__name = name
        self.__mode = mode
        self.__encoding = encoding
        self._read_buf = None
        self._writable_file = None
        self._binary_mode = 'b' in mode
        mode = mode.replace('b', '')
        if mode not in ('r', 'w', 'a', 'r+', 'w+', 'a+'):
            raise errors.InvalidArgumentError(None, None, "mode is not 'r' or 'w' or 'a' or 'r+' or 'w+' or 'a+'")
        self._read_check_passed = mode in ('r', 'r+', 'a+', 'w+')
        self._write_check_passed = mode in ('a', 'w', 'r+', 'a+', 'w+')

    @property
    def name(self):
        """Returns the file name."""
        return self.__name

    @property
    def mode(self):
        """Returns the mode in which the file was opened."""
        return self.__mode

    def _preread_check(self):
        if not self._read_buf:
            if not self._read_check_passed:
                raise errors.PermissionDeniedError(None, None, "File isn't open for reading")
            self._read_buf = _pywrap_file_io.BufferedInputStream(compat.path_to_str(self.__name), 1024 * 512)

    def _prewrite_check(self):
        if not self._writable_file:
            if not self._write_check_passed:
                raise errors.PermissionDeniedError(None, None, "File isn't open for writing")
            self._writable_file = _pywrap_file_io.WritableFile(compat.path_to_bytes(self.__name), compat.as_bytes(self.__mode))

    def _prepare_value(self, val):
        if self._binary_mode:
            return compat.as_bytes(val, encoding=self.__encoding)
        else:
            return compat.as_str_any(val, encoding=self.__encoding)

    def size(self):
        """Returns the size of the file."""
        return stat(self.__name).length

    def write(self, file_content):
        """Writes file_content to the file. Appends to the end of the file."""
        self._prewrite_check()
        self._writable_file.append(compat.as_bytes(file_content, encoding=self.__encoding))

    def read(self, n=-1):
        """Returns the contents of a file as a string.

    Starts reading from current position in file.

    Args:
      n: Read `n` bytes if `n != -1`. If `n = -1`, reads to end of file.

    Returns:
      `n` bytes of the file (or whole file) in bytes mode or `n` bytes of the
      string if in string (regular) mode.
    """
        self._preread_check()
        if n == -1:
            length = self.size() - self.tell()
        else:
            length = n
        return self._prepare_value(self._read_buf.read(length))

    @deprecation.deprecated_args(None, 'position is deprecated in favor of the offset argument.', 'position')
    def seek(self, offset=None, whence=0, position=None):
        """Seeks to the offset in the file.

    Args:
      offset: The byte count relative to the whence argument.
      whence: Valid values for whence are:
        0: start of the file (default)
        1: relative to the current position of the file
        2: relative to the end of file. `offset` is usually negative.
    """
        self._preread_check()
        if offset is None and position is None:
            raise TypeError('seek(): offset argument required')
        if offset is not None and position is not None:
            raise TypeError('seek(): offset and position may not be set simultaneously.')
        if position is not None:
            offset = position
        if whence == 0:
            pass
        elif whence == 1:
            offset += self.tell()
        elif whence == 2:
            offset += self.size()
        else:
            raise errors.InvalidArgumentError(None, None, 'Invalid whence argument: {}. Valid values are 0, 1, or 2.'.format(whence))
        self._read_buf.seek(offset)

    def readline(self):
        """Reads the next line, keeping \\n. At EOF, returns ''."""
        self._preread_check()
        return self._prepare_value(self._read_buf.readline())

    def readlines(self):
        """Returns all lines from the file in a list."""
        self._preread_check()
        lines = []
        while True:
            s = self.readline()
            if not s:
                break
            lines.append(s)
        return lines

    def tell(self):
        """Returns the current position in the file."""
        if self._read_check_passed:
            self._preread_check()
            return self._read_buf.tell()
        else:
            self._prewrite_check()
            return self._writable_file.tell()

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.readline()
        if not retval:
            raise StopIteration()
        return retval

    def next(self):
        return self.__next__()

    def flush(self):
        """Flushes the Writable file.

    This only ensures that the data has made its way out of the process without
    any guarantees on whether it's written to disk. This means that the
    data would survive an application crash but not necessarily an OS crash.
    """
        if self._writable_file:
            self._writable_file.flush()

    def close(self):
        """Closes the file.

    Should be called for the WritableFile to be flushed.

    In general, if you use the context manager pattern, you don't need to call
    this directly.

    >>> with tf.io.gfile.GFile("/tmp/x", "w") as f:
    ...   f.write("asdf\\n")
    ...   f.write("qwer\\n")
    >>> # implicit f.close() at the end of the block

    For cloud filesystems, forgetting to call `close()` might result in data
    loss as last write might not have been replicated.
    """
        self._read_buf = None
        if self._writable_file:
            self._writable_file.close()
            self._writable_file = None

    def seekable(self):
        """Returns True as FileIO supports random access ops of seek()/tell()"""
        return True