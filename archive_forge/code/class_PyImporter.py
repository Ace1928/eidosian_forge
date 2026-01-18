imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
class PyImporter(PyxImporter):
    """A meta-path importer for normal .py files.
    """

    def __init__(self, pyxbuild_dir=None, inplace=False, language_level=None):
        if language_level is None:
            language_level = sys.version_info[0]
        self.super = super(PyImporter, self)
        self.super.__init__(extension='.py', pyxbuild_dir=pyxbuild_dir, inplace=inplace, language_level=language_level)
        self.uncompilable_modules = {}
        self.blocked_modules = ['Cython', 'pyxbuild', 'pyximport.pyxbuild', 'distutils']
        self.blocked_packages = ['Cython.', 'distutils.']

    def find_module(self, fullname, package_path=None):
        if fullname in sys.modules:
            return None
        if any([fullname.startswith(pkg) for pkg in self.blocked_packages]):
            return None
        if fullname in self.blocked_modules:
            return None
        if _lib_loader.knows(fullname):
            return _lib_loader
        _debug("trying import of module '%s'", fullname)
        if fullname in self.uncompilable_modules:
            path, last_modified = self.uncompilable_modules[fullname]
            try:
                new_last_modified = os.stat(path).st_mtime
                if new_last_modified > last_modified:
                    return None
            except OSError:
                pass
        self.blocked_modules.append(fullname)
        try:
            importer = self.super.find_module(fullname, package_path)
            if importer is not None:
                if importer.init_path:
                    path = importer.init_path
                    real_name = fullname + '.__init__'
                else:
                    path = importer.path
                    real_name = fullname
                _debug('importer found path %s for module %s', path, real_name)
                try:
                    so_path = build_module(real_name, path, pyxbuild_dir=self.pyxbuild_dir, language_level=self.language_level, inplace=self.inplace)
                    _lib_loader.add_lib(fullname, path, so_path, is_package=bool(importer.init_path))
                    return _lib_loader
                except Exception:
                    if DEBUG_IMPORT:
                        import traceback
                        traceback.print_exc()
                    try:
                        last_modified = os.stat(path).st_mtime
                    except OSError:
                        last_modified = 0
                    self.uncompilable_modules[fullname] = (path, last_modified)
                    importer = None
        finally:
            self.blocked_modules.pop()
        return importer