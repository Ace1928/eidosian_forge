import _frozen_importlib_external as _bootstrap_external
from _frozen_importlib_external import _unpack_uint16, _unpack_uint32
import _frozen_importlib as _bootstrap  # for _verbose_message
import _imp  # for check_hash_based_pycs
import _io  # for open
import marshal  # for loads
import sys  # for modules
import time  # for mktime
import _warnings  # For warn()
class zipimporter(_bootstrap_external._LoaderBasics):
    """zipimporter(archivepath) -> zipimporter object

    Create a new zipimporter instance. 'archivepath' must be a path to
    a zipfile, or to a specific path inside a zipfile. For example, it can be
    '/tmp/myimport.zip', or '/tmp/myimport.zip/mydirectory', if mydirectory is a
    valid directory inside the archive.

    'ZipImportError is raised if 'archivepath' doesn't point to a valid Zip
    archive.

    The 'archive' attribute of zipimporter objects contains the name of the
    zipfile targeted.
    """

    def __init__(self, path):
        if not isinstance(path, str):
            raise TypeError(f'expected str, not {type(path)!r}')
        if not path:
            raise ZipImportError('archive path is empty', path=path)
        if alt_path_sep:
            path = path.replace(alt_path_sep, path_sep)
        prefix = []
        while True:
            try:
                st = _bootstrap_external._path_stat(path)
            except (OSError, ValueError):
                dirname, basename = _bootstrap_external._path_split(path)
                if dirname == path:
                    raise ZipImportError('not a Zip file', path=path)
                path = dirname
                prefix.append(basename)
            else:
                if st.st_mode & 61440 != 32768:
                    raise ZipImportError('not a Zip file', path=path)
                break
        try:
            files = _zip_directory_cache[path]
        except KeyError:
            files = _read_directory(path)
            _zip_directory_cache[path] = files
        self._files = files
        self.archive = path
        self.prefix = _bootstrap_external._path_join(*prefix[::-1])
        if self.prefix:
            self.prefix += path_sep

    def find_loader(self, fullname, path=None):
        """find_loader(fullname, path=None) -> self, str or None.

        Search for a module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the zipimporter
        instance itself if the module was found, a string containing the
        full path name if it's possibly a portion of a namespace package,
        or None otherwise. The optional 'path' argument is ignored -- it's
        there for compatibility with the importer protocol.

        Deprecated since Python 3.10. Use find_spec() instead.
        """
        _warnings.warn('zipimporter.find_loader() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        mi = _get_module_info(self, fullname)
        if mi is not None:
            return (self, [])
        modpath = _get_module_path(self, fullname)
        if _is_dir(self, modpath):
            return (None, [f'{self.archive}{path_sep}{modpath}'])
        return (None, [])

    def find_module(self, fullname, path=None):
        """find_module(fullname, path=None) -> self or None.

        Search for a module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the zipimporter
        instance itself if the module was found, or None if it wasn't.
        The optional 'path' argument is ignored -- it's there for compatibility
        with the importer protocol.

        Deprecated since Python 3.10. Use find_spec() instead.
        """
        _warnings.warn('zipimporter.find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        return self.find_loader(fullname, path)[0]

    def find_spec(self, fullname, target=None):
        """Create a ModuleSpec for the specified module.

        Returns None if the module cannot be found.
        """
        module_info = _get_module_info(self, fullname)
        if module_info is not None:
            return _bootstrap.spec_from_loader(fullname, self, is_package=module_info)
        else:
            modpath = _get_module_path(self, fullname)
            if _is_dir(self, modpath):
                path = f'{self.archive}{path_sep}{modpath}'
                spec = _bootstrap.ModuleSpec(name=fullname, loader=None, is_package=True)
                spec.submodule_search_locations.append(path)
                return spec
            else:
                return None

    def get_code(self, fullname):
        """get_code(fullname) -> code object.

        Return the code object for the specified module. Raise ZipImportError
        if the module couldn't be imported.
        """
        code, ispackage, modpath = _get_module_code(self, fullname)
        return code

    def get_data(self, pathname):
        """get_data(pathname) -> string with file data.

        Return the data associated with 'pathname'. Raise OSError if
        the file wasn't found.
        """
        if alt_path_sep:
            pathname = pathname.replace(alt_path_sep, path_sep)
        key = pathname
        if pathname.startswith(self.archive + path_sep):
            key = pathname[len(self.archive + path_sep):]
        try:
            toc_entry = self._files[key]
        except KeyError:
            raise OSError(0, '', key)
        return _get_data(self.archive, toc_entry)

    def get_filename(self, fullname):
        """get_filename(fullname) -> filename string.

        Return the filename for the specified module or raise ZipImportError
        if it couldn't be imported.
        """
        code, ispackage, modpath = _get_module_code(self, fullname)
        return modpath

    def get_source(self, fullname):
        """get_source(fullname) -> source string.

        Return the source code for the specified module. Raise ZipImportError
        if the module couldn't be found, return None if the archive does
        contain the module, but has no source for it.
        """
        mi = _get_module_info(self, fullname)
        if mi is None:
            raise ZipImportError(f"can't find module {fullname!r}", name=fullname)
        path = _get_module_path(self, fullname)
        if mi:
            fullpath = _bootstrap_external._path_join(path, '__init__.py')
        else:
            fullpath = f'{path}.py'
        try:
            toc_entry = self._files[fullpath]
        except KeyError:
            return None
        return _get_data(self.archive, toc_entry).decode()

    def is_package(self, fullname):
        """is_package(fullname) -> bool.

        Return True if the module specified by fullname is a package.
        Raise ZipImportError if the module couldn't be found.
        """
        mi = _get_module_info(self, fullname)
        if mi is None:
            raise ZipImportError(f"can't find module {fullname!r}", name=fullname)
        return mi

    def load_module(self, fullname):
        """load_module(fullname) -> module.

        Load the module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the imported
        module, or raises ZipImportError if it could not be imported.

        Deprecated since Python 3.10. Use exec_module() instead.
        """
        msg = 'zipimport.zipimporter.load_module() is deprecated and slated for removal in Python 3.12; use exec_module() instead'
        _warnings.warn(msg, DeprecationWarning)
        code, ispackage, modpath = _get_module_code(self, fullname)
        mod = sys.modules.get(fullname)
        if mod is None or not isinstance(mod, _module_type):
            mod = _module_type(fullname)
            sys.modules[fullname] = mod
        mod.__loader__ = self
        try:
            if ispackage:
                path = _get_module_path(self, fullname)
                fullpath = _bootstrap_external._path_join(self.archive, path)
                mod.__path__ = [fullpath]
            if not hasattr(mod, '__builtins__'):
                mod.__builtins__ = __builtins__
            _bootstrap_external._fix_up_module(mod.__dict__, fullname, modpath)
            exec(code, mod.__dict__)
        except:
            del sys.modules[fullname]
            raise
        try:
            mod = sys.modules[fullname]
        except KeyError:
            raise ImportError(f'Loaded module {fullname!r} not found in sys.modules')
        _bootstrap._verbose_message('import {} # loaded from Zip {}', fullname, modpath)
        return mod

    def get_resource_reader(self, fullname):
        """Return the ResourceReader for a package in a zip file.

        If 'fullname' is a package within the zip file, return the
        'ResourceReader' object for the package.  Otherwise return None.
        """
        try:
            if not self.is_package(fullname):
                return None
        except ZipImportError:
            return None
        from importlib.readers import ZipReader
        return ZipReader(self, fullname)

    def invalidate_caches(self):
        """Reload the file data of the archive path."""
        try:
            self._files = _read_directory(self.archive)
            _zip_directory_cache[self.archive] = self._files
        except ZipImportError:
            _zip_directory_cache.pop(self.archive, None)
            self._files = {}

    def __repr__(self):
        return f'<zipimporter object "{self.archive}{path_sep}{self.prefix}">'