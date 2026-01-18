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
class TempfileManagerClass(object):
    """A class for managing tempfile contexts

    Pyomo declares a global instance of this class as ``TempfileManager``:

    .. doctest::

       >>> from pyomo.common.tempfiles import TempfileManager

    This class provides an interface for managing
    :class:`TempfileContext` contexts.  It implements a basic stack,
    where users can :meth:`push()` a new context (causing it to become
    the current "active" context) and :meth:`pop()` contexts off
    (optionally deleting all files associated with the context).  In
    general usage, users will either use this class to create new
    tempfile contexts and use them explicitly (i.e., through a context
    manager):

    .. doctest::

       >>> import os
       >>> with TempfileManager.new_context() as tempfile:
       ...     fd, fname = tempfile.mkstemp()
       ...     dname = tempfile.mkdtemp()
       ...     os.path.isfile(fname)
       ...     os.path.isdir(dname)
       True
       True
       >>> os.path.exists(fname)
       False
       >>> os.path.exists(dname)
       False

    or through an implicit active context accessed through the manager
    class:

    .. doctest::

       >>> TempfileManager.push()
       <pyomo.common.tempfiles.TempfileContext object ...>
       >>> fname = TempfileManager.create_tempfile()
       >>> dname = TempfileManager.create_tempdir()
       >>> os.path.isfile(fname)
       True
       >>> os.path.isdir(dname)
       True

       >>> TempfileManager.pop()
       <pyomo.common.tempfiles.TempfileContext object ...>
       >>> os.path.exists(fname)
       False
       >>> os.path.exists(dname)
       False

    """

    def __init__(self):
        self._context_stack = []
        self._context_manager_stack = []
        self.tempdir = None

    def __del__(self):
        self.shutdown()

    def shutdown(self, remove=True):
        if not self._context_stack:
            return
        if any((ctx.tempfiles for ctx in self._context_stack)):
            logger.error('Temporary files created through TempfileManager contexts have not been deleted (observed during TempfileManager instance shutdown).\nUndeleted entries:\n\t' + '\n\t'.join((fname if isinstance(fname, str) else fname.decode() for ctx in self._context_stack for fd, fname in ctx.tempfiles)))
        if self._context_stack:
            logger.warning('TempfileManagerClass instance: un-popped tempfile contexts still exist during TempfileManager instance shutdown')
        self.clear_tempfiles(remove)
        self._context_stack = None

    def context(self):
        """Return the current active TempfileContext.

        Raises
        ------
        TempfileContextError if there is not a current context."""
        if not self._context_stack:
            raise TempfileContextError('TempfileManager has no currently active context.  Create a context (with push() or __enter__()) before attempting to create temporary objects.')
        return self._context_stack[-1]

    def create_tempfile(self, suffix=None, prefix=None, text=False, dir=None):
        """Call :meth:`TempfileContext.create_tempfile` on the active context"""
        return self.context().create_tempfile(suffix=suffix, prefix=prefix, text=text, dir=dir)

    def create_tempdir(self, suffix=None, prefix=None, dir=None):
        """Call :meth:`TempfileContext.create_tempdir` on the active context"""
        return self.context().create_tempdir(suffix=suffix, prefix=prefix, dir=dir)

    def add_tempfile(self, filename, exists=True):
        """Call :meth:`TempfileContext.add_tempfile` on the active context"""
        return self.context().add_tempfile(filename=filename, exists=exists)

    def clear_tempfiles(self, remove=True):
        """Delete all temporary files and remove all contexts."""
        while self._context_stack:
            self.pop(remove)

    @deprecated('The TempfileManager.sequential_files() method has been removed.  All temporary files are created with guaranteed unique names.  Users wishing sequentially numbered files should create a temporary (empty) directory using mkdtemp / create_tempdir and place the sequential files within it.', version='6.2')
    def sequential_files(self, ctr=0):
        pass

    def unique_files(self):
        pass

    def new_context(self):
        """Create and return an new tempfile context

        Returns
        -------
        TempfileContext
            the newly-created tempfile context

        """
        return TempfileContext(self)

    def push(self):
        """Create a new tempfile context and set it as the active context.

        Returns
        -------
        TempfileContext
            the newly-created tempfile context

        """
        context = self.new_context()
        self._context_stack.append(context)
        return context

    def pop(self, remove=True):
        """Remove and release the active context

        Parameters
        ----------
        remove: bool
            If ``True``, delete all managed files / directories

        """
        ctx = self._context_stack.pop()
        ctx.release(remove)
        return ctx

    def __enter__(self):
        ctx = self.push()
        self._context_manager_stack.append(ctx)
        return ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        ctx = self._context_manager_stack.pop()
        while True:
            if ctx is self.pop():
                break
            logger.warning('TempfileManager: tempfile context was pushed onto the TempfileManager stack within a context manager (i.e., `with TempfileManager:`) but was not popped before the context manager exited.  Popping the context to preserve the stack integrity.')