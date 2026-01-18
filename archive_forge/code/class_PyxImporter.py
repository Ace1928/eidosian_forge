imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
class PyxImporter(object):
    """A meta-path importer for .pyx files.
    """

    def __init__(self, extension=PYX_EXT, pyxbuild_dir=None, inplace=False, language_level=None):
        self.extension = extension
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level

    def find_module(self, fullname, package_path=None):
        if fullname in sys.modules and (not pyxargs.reload_support):
            return None
        if package_path is not None and (not isinstance(package_path, list)):
            package_path = list(package_path)
        try:
            fp, pathname, (ext, mode, ty) = imp.find_module(fullname, package_path)
            if fp:
                fp.close()
            if pathname and ty == imp.PKG_DIRECTORY:
                pkg_file = os.path.join(pathname, '__init__' + self.extension)
                if os.path.isfile(pkg_file):
                    return PyxLoader(fullname, pathname, init_path=pkg_file, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
            if pathname and pathname.endswith(self.extension):
                return PyxLoader(fullname, pathname, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
            if ty != imp.C_EXTENSION:
                return None
            pyxpath = os.path.splitext(pathname)[0] + self.extension
            if os.path.isfile(pyxpath):
                return PyxLoader(fullname, pyxpath, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
        except ImportError:
            pass
        mod_parts = fullname.split('.')
        module_name = mod_parts[-1]
        pyx_module_name = module_name + self.extension
        paths = package_path or sys.path
        for path in paths:
            pyx_data = None
            if not path:
                path = os.getcwd()
            elif os.path.isfile(path):
                try:
                    zi = zipimporter(path)
                    pyx_data = zi.get_data(pyx_module_name)
                except (ZipImportError, IOError, OSError):
                    continue
                path = self.pyxbuild_dir
            elif not os.path.isabs(path):
                path = os.path.abspath(path)
            pyx_module_path = os.path.join(path, pyx_module_name)
            if pyx_data is not None:
                if not os.path.exists(path):
                    try:
                        os.makedirs(path)
                    except OSError:
                        if not os.path.exists(path):
                            raise
                with open(pyx_module_path, 'wb') as f:
                    f.write(pyx_data)
            elif not os.path.isfile(pyx_module_path):
                continue
            return PyxLoader(fullname, pyx_module_path, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
        _debug('%s not found' % fullname)
        return None