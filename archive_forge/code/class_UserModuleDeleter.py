import sys
import os
from _pydev_bundle._pydev_execfile import execfile
class UserModuleDeleter:
    """
    User Module Deleter (UMD) aims at deleting user modules
    to force Python to deeply reload them during import

    pathlist [list]: ignore list in terms of module path
    namelist [list]: ignore list in terms of module name
    """

    def __init__(self, namelist=None, pathlist=None):
        if namelist is None:
            namelist = []
        self.namelist = namelist
        if pathlist is None:
            pathlist = []
        self.pathlist = pathlist
        try:
            import pydev_pysrc, inspect
            self.pathlist.append(os.path.dirname(pydev_pysrc.__file__))
        except:
            pass
        self.previous_modules = list(sys.modules.keys())

    def is_module_ignored(self, modname, modpath):
        for path in [sys.prefix] + self.pathlist:
            if modpath.startswith(path):
                return True
        else:
            return set(modname.split('.')) & set(self.namelist)

    def run(self, verbose=False):
        """
        Del user modules to force Python to deeply reload them

        Do not del modules which are considered as system modules, i.e.
        modules installed in subdirectories of Python interpreter's binary
        Do not del C modules
        """
        log = []
        modules_copy = dict(sys.modules)
        for modname, module in modules_copy.items():
            if modname == 'aaaaa':
                print(modname, module)
                print(self.previous_modules)
            if modname not in self.previous_modules:
                modpath = getattr(module, '__file__', None)
                if modpath is None:
                    continue
                if not self.is_module_ignored(modname, modpath):
                    log.append(modname)
                    del sys.modules[modname]
        if verbose and log:
            print('\x1b[4;33m%s\x1b[24m%s\x1b[0m' % ('UMD has deleted', ': ' + ', '.join(log)))