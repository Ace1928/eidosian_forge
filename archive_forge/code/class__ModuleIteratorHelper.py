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
class _ModuleIteratorHelper:
    """
    This mixin provides common behavior between python module and path entries,
    since the mechanism for searching sys.path and __path__ attributes is
    remarkably similar.
    """

    def iterModules(self):
        """
        Loop over the modules present below this entry or package on PYTHONPATH.

        For modules which are not packages, this will yield nothing.

        For packages and path entries, this will only yield modules one level
        down; i.e. if there is a package a.b.c, iterModules on a will only
        return a.b.  If you want to descend deeply, use walkModules.

        @return: a generator which yields PythonModule instances that describe
        modules which can be, or have been, imported.
        """
        yielded = {}
        if not self.filePath.exists():
            return
        for placeToLook in self._packagePaths():
            try:
                children = sorted(placeToLook.children())
            except UnlistableError:
                continue
            for potentialTopLevel in children:
                ext = potentialTopLevel.splitext()[1]
                potentialBasename = potentialTopLevel.basename()[:-len(ext)]
                if ext in PYTHON_EXTENSIONS:
                    if not _isPythonIdentifier(potentialBasename):
                        continue
                    modname = self._subModuleName(potentialBasename)
                    if modname.split('.')[-1] == '__init__':
                        continue
                    if modname not in yielded:
                        yielded[modname] = True
                        pm = PythonModule(modname, potentialTopLevel, self._getEntry())
                        assert pm != self
                        yield pm
                else:
                    if ext or not _isPythonIdentifier(potentialBasename) or (not potentialTopLevel.isdir()):
                        continue
                    modname = self._subModuleName(potentialTopLevel.basename())
                    for ext in PYTHON_EXTENSIONS:
                        initpy = potentialTopLevel.child('__init__' + ext)
                        if initpy.exists() and modname not in yielded:
                            yielded[modname] = True
                            pm = PythonModule(modname, initpy, self._getEntry())
                            assert pm != self
                            yield pm
                            break

    def walkModules(self, importPackages=False):
        """
        Similar to L{iterModules}, this yields self, and then every module in my
        package or entry, and every submodule in each package or entry.

        In other words, this is deep, and L{iterModules} is shallow.
        """
        yield self
        for package in self.iterModules():
            yield from package.walkModules(importPackages=importPackages)

    def _subModuleName(self, mn):
        """
        This is a hook to provide packages with the ability to specify their names
        as a prefix to submodules here.
        """
        return mn

    def _packagePaths(self):
        """
        Implement in subclasses to specify where to look for modules.

        @return: iterable of FilePath-like objects.
        """
        raise NotImplementedError()

    def _getEntry(self):
        """
        Implement in subclasses to specify what path entry submodules will come
        from.

        @return: a PathEntry instance.
        """
        raise NotImplementedError()

    def __getitem__(self, modname):
        """
        Retrieve a module from below this path or package.

        @param modname: a str naming a module to be loaded.  For entries, this
        is a top-level, undotted package name, and for packages it is the name
        of the module without the package prefix.  For example, if you have a
        PythonModule representing the 'twisted' package, you could use::

            twistedPackageObj['python']['modules']

        to retrieve this module.

        @raise KeyError: if the module is not found.

        @return: a PythonModule.
        """
        for module in self.iterModules():
            if module.name == self._subModuleName(modname):
                return module
        raise KeyError(modname)

    def __iter__(self):
        """
        Implemented to raise NotImplementedError for clarity, so that attempting to
        loop over this object won't call __getitem__.

        Note: in the future there might be some sensible default for iteration,
        like 'walkEverything', so this is deliberately untested and undefined
        behavior.
        """
        raise NotImplementedError()