from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
class PythonPath:
    """
    I represent the very top of the Python object-space, the module list in
    C{sys.path} and the modules list in C{sys.modules}.

    @ivar _sysPath: A sequence of strings like C{sys.path}.  This attribute is
    read-only.

    @ivar sysPath: The current value of the module search path list.
    @type sysPath: C{list}

    @ivar moduleDict: A dictionary mapping string module names to module
    objects, like C{sys.modules}.

    @ivar sysPathHooks: A list of PEP-302 path hooks, like C{sys.path_hooks}.

    @ivar moduleLoader: A function that takes a fully-qualified python name and
    returns a module, like L{twisted.python.reflect.namedAny}.
    """

    def __init__(self, sysPath=None, moduleDict=sys.modules, sysPathHooks=sys.path_hooks, importerCache=sys.path_importer_cache, moduleLoader=namedAny, sysPathFactory=None):
        """
        Create a PythonPath.  You almost certainly want to use
        modules.theSystemPath, or its aliased methods, rather than creating a
        new instance yourself, though.

        All parameters are optional, and if unspecified, will use 'system'
        equivalents that makes this PythonPath like the global L{theSystemPath}
        instance.

        @param sysPath: a sys.path-like list to use for this PythonPath, to
        specify where to load modules from.

        @param moduleDict: a sys.modules-like dictionary to use for keeping
        track of what modules this PythonPath has loaded.

        @param sysPathHooks: sys.path_hooks-like list of PEP-302 path hooks to
        be used for this PythonPath, to determie which importers should be
        used.

        @param importerCache: a sys.path_importer_cache-like list of PEP-302
        importers.  This will be used in conjunction with the given
        sysPathHooks.

        @param moduleLoader: a module loader function which takes a string and
        returns a module.  That is to say, it is like L{namedAny} - *not* like
        L{__import__}.

        @param sysPathFactory: a 0-argument callable which returns the current
        value of a sys.path-like list of strings.  Specify either this, or
        sysPath, not both.  This alternative interface is provided because the
        way the Python import mechanism works, you can re-bind the 'sys.path'
        name and that is what is used for current imports, so it must be a
        factory rather than a value to deal with modification by rebinding
        rather than modification by mutation.  Note: it is not recommended to
        rebind sys.path.  Although this mechanism can deal with that, it is a
        subtle point which some tools that it is easy for tools which interact
        with sys.path to miss.
        """
        if sysPath is not None:
            sysPathFactory = lambda: sysPath
        elif sysPathFactory is None:
            sysPathFactory = _defaultSysPathFactory
        self._sysPathFactory = sysPathFactory
        self._sysPath = sysPath
        self.moduleDict = moduleDict
        self.sysPathHooks = sysPathHooks
        self.importerCache = importerCache
        self.moduleLoader = moduleLoader

    @property
    def sysPath(self):
        """
        Retrieve the current value of the module search path list.
        """
        return self._sysPathFactory()

    def _findEntryPathString(self, modobj):
        """
        Determine where a given Python module object came from by looking at path
        entries.
        """
        topPackageObj = modobj
        while '.' in topPackageObj.__name__:
            topPackageObj = self.moduleDict['.'.join(topPackageObj.__name__.split('.')[:-1])]
        if _isPackagePath(FilePath(topPackageObj.__file__)):
            rval = dirname(dirname(topPackageObj.__file__))
        else:
            rval = dirname(topPackageObj.__file__)
        if rval not in self.importerCache:
            warnings.warn('%s (for module %s) not in path importer cache (PEP 302 violation - check your local configuration).' % (rval, modobj.__name__), stacklevel=3)
        return rval

    def _smartPath(self, pathName):
        """
        Given a path entry from sys.path which may refer to an importer,
        return the appropriate FilePath-like instance.

        @param pathName: a str describing the path.

        @return: a FilePath-like object.
        """
        importr = self.importerCache.get(pathName, _nothing)
        if importr is _nothing:
            for hook in self.sysPathHooks:
                try:
                    importr = hook(pathName)
                except ImportError:
                    pass
            if importr is _nothing:
                importr = None
        return IPathImportMapper(importr, _theDefaultMapper).mapPath(pathName)

    def iterEntries(self):
        """
        Iterate the entries on my sysPath.

        @return: a generator yielding PathEntry objects
        """
        for pathName in self.sysPath:
            fp = self._smartPath(pathName)
            yield PathEntry(fp, self)

    def __getitem__(self, modname):
        """
        Get a python module by its given fully-qualified name.

        @param modname: The fully-qualified Python module name to load.

        @type modname: C{str}

        @return: an object representing the module identified by C{modname}

        @rtype: L{PythonModule}

        @raise KeyError: if the module name is not a valid module name, or no
            such module can be identified as loadable.
        """
        moduleObject = self.moduleDict.get(modname)
        if moduleObject is not None:
            pe = PathEntry(self._smartPath(self._findEntryPathString(moduleObject)), self)
            mp = self._smartPath(moduleObject.__file__)
            return PythonModule(modname, mp, pe)
        if '.' in modname:
            pkg = self
            for name in modname.split('.'):
                pkg = pkg[name]
            return pkg
        for module in self.iterModules():
            if module.name == modname:
                return module
        raise KeyError(modname)

    def __contains__(self, module):
        """
        Check to see whether or not a module exists on my import path.

        @param module: The name of the module to look for on my import path.
        @type module: C{str}
        """
        try:
            self.__getitem__(module)
            return True
        except KeyError:
            return False

    def __repr__(self) -> str:
        """
        Display my sysPath and moduleDict in a string representation.
        """
        return f'PythonPath({self.sysPath!r},{self.moduleDict!r})'

    def iterModules(self):
        """
        Yield all top-level modules on my sysPath.
        """
        for entry in self.iterEntries():
            yield from entry.iterModules()

    def walkModules(self, importPackages=False):
        """
        Similar to L{iterModules}, this yields every module on the path, then every
        submodule in each package or entry.
        """
        for package in self.iterModules():
            yield from package.walkModules(importPackages=False)