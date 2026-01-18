import os
import time
import tempfile
import logging
import shutil
import weakref
from pyomo.common.dependencies import attempt_import, pyutilib_available
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TempfileContextError
from pyomo.common.multithread import MultiThreadWrapperWithMain
class TempfileContext:
    """A `context` for managing collections of temporary files

    Instances of this class hold a "temporary file context".  That is,
    this records a collection of temporary file system objects that are
    all managed as a group.  The most common use of the context is to
    ensure that all files are deleted when the context is released.

    This class replicates a significant portion of the :mod:`tempfile`
    module interface.

    Instances of this class may be used as context managers (with the
    temporary files / directories getting automatically deleted when the
    context manager exits).

    Instances will also attempt to delete any temporary objects from the
    filesystem when the context falls out of scope (although this
    behavior is not guaranteed for instances existing when the
    interpreter is shutting down).

    """

    def __init__(self, manager):
        self.manager = weakref.ref(manager)
        self.tempfiles = []
        self.tempdir = None

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def mkstemp(self, suffix=None, prefix=None, dir=None, text=False):
        """Create a unique temporary file using :func:`tempfile.mkstemp`

        Parameters are handled as in :func:`tempfile.mkstemp`, with
        the exception that the new file is created in the directory
        returned by :meth:`gettempdir`

        Returns
        -------
        fd: int
            the opened file descriptor

        fname: str or bytes
            the absolute path to the new temporary file

        """
        dir = self._resolve_tempdir(dir)
        ans = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        self.tempfiles.append(ans)
        return ans

    def mkdtemp(self, suffix=None, prefix=None, dir=None):
        """Create a unique temporary directory using :func:`tempfile.mkdtemp`

        Parameters are handled as in :func:`tempfile.mkdtemp`, with
        the exception that the new file is created in the directory
        returned by :meth:`gettempdir`

        Returns
        -------
        dname: str or bytes
            the absolute path to the new temporary directory

        """
        dir = self._resolve_tempdir(dir)
        dname = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self.tempfiles.append((None, dname))
        return dname

    def gettempdir(self):
        """Return the default name of the directory used for temporary files.

        This method returns the first non-null location returned from:

         - This context's ``tempdir`` (i.e., ``self.tempdir``)
         - This context's manager's ``tempdir`` (i.e.,
           ``self.manager().tempdir``)
         - :func:`tempfile.gettempdir()`

        Returns
        -------
        dir: str
            The default directory to use for creating temporary objects
        """
        dir = self._resolve_tempdir()
        if dir is None:
            return tempfile.gettempdir()
        if isinstance(dir, bytes):
            return dir.decode()
        return dir

    def gettempdirb(self):
        """Same as :meth:`gettempdir()`, but the return value is ``bytes``"""
        dir = self._resolve_tempdir()
        if dir is None:
            return tempfile.gettempdirb()
        if not isinstance(dir, bytes):
            return dir.encode()
        return dir

    def gettempprefix(self):
        """Return the filename prefix used to create temporary files.

        See :func:`tempfile.gettempprefix()`

        """
        return tempfile.gettempprefix()

    def gettempprefixb(self):
        """Same as :meth:`gettempprefix()`, but the return value is ``bytes``"""
        return tempfile.gettempprefixb()

    def create_tempfile(self, suffix=None, prefix=None, text=False, dir=None):
        """Create a unique temporary file.

        The file name is generated as in :func:`tempfile.mkstemp()`.

        Any file handles to the new file (e.g., from :meth:`mkstemp`)
        are closed.

        Returns
        -------
        fname: str or bytes
            The absolute path of the new file.

        """
        fd, fname = self.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        os.close(fd)
        self.tempfiles[-1] = (None, fname)
        return fname

    def create_tempdir(self, suffix=None, prefix=None, dir=None):
        """Create a unique temporary directory.

        The file name is generated as in :func:`tempfile.mkdtemp()`.

        Returns
        -------
        dname: str or bytes
            The absolute path of the new directory.

        """
        return self.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    def add_tempfile(self, filename, exists=True):
        """Declare the specified file/directory to be temporary.

        This adds the specified path as a "temporary" object to this
        context's list of managed temporary paths (i.e., it will be
        potentially be deleted when the context is released (see
        :meth:`release`).

        Parameters
        ----------
        filename: str
            the file / directory name to be treated as temporary
        exists: bool
            if ``True``, the file / directory must already exist.

        """
        tmp = os.path.abspath(filename)
        if exists and (not os.path.exists(tmp)):
            raise IOError('Temporary file does not exist: ' + tmp)
        self.tempfiles.append((None, tmp))

    def release(self, remove=True):
        """Release this context

        This releases the current context, potentially deleting all
        managed temporary objects (files and directories), and resetting
        the context to generate unique names.

        Parameters
        ----------
        remove: bool
            If ``True``, delete all managed files / directories
        """
        if remove:
            for fd, name in reversed(self.tempfiles):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                self._remove_filesystem_object(name)
        self.tempfiles.clear()

    def _resolve_tempdir(self, dir=None):
        if dir is not None:
            return dir
        elif self.tempdir is not None:
            return self.tempdir
        elif self.manager().tempdir is not None:
            return self.manager().tempdir
        elif TempfileManager.main_thread.tempdir is not None:
            return TempfileManager.main_thread.tempdir
        elif pyutilib_available:
            if pyutilib_tempfiles.TempfileManager.tempdir is not None:
                deprecation_warning('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files has been deprecated.  Please set TempfileManager.tempdir in pyomo.common.tempfiles', version='5.7.2')
                return pyutilib_tempfiles.TempfileManager.tempdir
        return None

    def _remove_filesystem_object(self, name):
        if not os.path.exists(name):
            return
        if os.path.isfile(name) or os.path.islink(name):
            try:
                os.remove(name)
            except WindowsError:
                try:
                    time.sleep(1)
                    os.remove(name)
                except WindowsError:
                    if deletion_errors_are_fatal:
                        raise
                    else:
                        logger = logging.getLogger(__name__)
                        logger.warning('Unable to delete temporary file %s' % (name,))
            return
        assert os.path.isdir(name)
        shutil.rmtree(name, ignore_errors=not deletion_errors_are_fatal)