import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
class PathManager(object):
    """The PathManager defines a registry class for path locations

    The :py:class:`PathManager` defines a class very similar to the
    :py:class:`CachedFactory` class; however it does not register type
    constructors.  Instead, it registers instances of
    :py:class:`PathData` (or :py:class:`ExecutableData`).  These
    contain the resolved path to the directory object under which the
    :py:class:`PathData` object was registered.  We do not use
    the PyUtilib ``register_executable`` and ``registered_executable``
    functions so that we can automatically include Pyomo-specific
    locations in the search path (namely the ``PYOMO_CONFIG_DIR``).

    Users will generally interact with this class through global
    instances of this class (``pyomo.common.Executable`` and
    ``pyomo.common.Library``).

    Users are not required or expected to register file names with the
    :py:class:`PathManager`; they will be automatically registered
    upon first use.  Generally, users interact through the ``path()``
    and ``available()`` methods:

    .. doctest::
        :hide:

        >>> import pyomo.common
        >>> import os
        >>> from stat import S_IXUSR, S_IXGRP, S_IXOTH
        >>> _testfile = os.path.join(
        ...    pyomo.common.envvar.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file')
        >>> _del_testfile = not os.path.exists(_testfile)
        >>> if _del_testfile:
        ...     open(_testfile,'w').close()
        ...     _mode = os.stat(_testfile).st_mode
        ...     os.chmod(_testfile, _mode | S_IXUSR | S_IXGRP | S_IXOTH)

    .. doctest::

        >>> from pyomo.common import Executable
        >>> if Executable('demo_exec_file').available():
        ...     loc = Executable('demo_exec_file').path()
        ...     print(os.path.isfile(loc))
        True
        >>> print(os.access(loc, os.X_OK))
        True

    For convenience, :py:meth:`available()` and :py:meth:`path()` are
    available by casting the :py:class:`PathData` object requrned
    from ``Executable`` or ``Library`` to either a ``bool`` or ``str``:

    .. doctest::

        >>> if Executable('demo_exec_file'):
        ...     cmd = "%s --help" % Executable('demo_exec_file')

    The :py:class:`PathManager` caches the location / existence of
    the target directory entry.  If something in the environment changes
    (e.g., the PATH) or the file is created or removed after the first
    time a client queried the location or availability, the
    PathManager will return incorrect information.  You can cause
    the :py:class:`PathManager` to refresh its cache by calling
    ``rehash()`` on either the :py:class:`PathData` (for the
    single file) or the :py:class:`PathManager` to refresh the
    cache for all files:

    .. doctest::

        >>> # refresh the cache for a single file
        >>> Executable('demo_exec_file').rehash()
        >>> # or all registered files
        >>> Executable.rehash()

    The ``Executable`` singleton looks for executables in the system
    ``PATH`` and in the list of directories specified by the ``pathlist``
    attribute.  ``Executable.pathlist`` defaults to a list containing the
    ``os.path.join(pyomo.common.envvar.PYOMO_CONFIG_DIR, 'bin')``.

    The ``Library`` singleton looks for executables in the system
    ``LD_LIBRARY_PATH``, ``PATH`` and in the list of directories
    specified by the ``pathlist`` attribute.  ``Library.pathlist``
    defaults to a list containing the
    ``os.path.join(pyomo.common.envvar.PYOMO_CONFIG_DIR, 'lib')``.

    Users may also override the normal file resolution by explicitly
    setting the location using :py:meth:`set_path`:

    .. doctest::

        >>> Executable('demo_exec_file').set_path(os.path.join(
        ...     pyomo.common.envvar.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file'))

    Explicitly setting the path is an absolute operation and will
    set the location whether or not that location points to an actual
    file.  Additionally, the explicit location will persist
    through calls to ``rehash()``.  If you wish to remove the explicit
    executable location, call ``set_path(None)``:

    .. doctest::

        >>> Executable('demo_exec_file').set_path(None)

    The ``Executable`` singleton uses :py:class:`ExecutableData`, an
    extended form of the :py:class:`PathData` class, which provides the
    ``executable`` property as an alais for :py:meth:`path()` and
    :py:meth:`set_path()`:

    .. doctest::

        >>> loc = Executable('demo_exec_file').executable
        >>> print(os.path.isfile(loc))
        True
        >>> Executable('demo_exec_file').executable = os.path.join(
        ...     pyomo.common.envvar.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file')
        >>> Executable('demo_exec_file').executable = None

    .. doctest::
        :hide:

        >>> if _del_testfile:
        ...     os.remove(_testfile)

    """

    def __init__(self, finder, dataClass):
        self._pathTo = {}
        self._find = finder
        self._dataClass = dataClass
        self.pathlist = None

    def __call__(self, path):
        if path not in self._pathTo:
            self._pathTo[path] = self._dataClass(self, path)
        return self._pathTo[path]

    def rehash(self):
        """Requery the location of all registered executables

        This method derives its name from the csh command of the same
        name, which rebuilds the hash table of executables reachable
        through the PATH.

        """
        for _path in self._pathTo.values():
            _path.rehash()