imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
class PyxImportLoader(ExtensionFileLoader):

    def __init__(self, filename, pyxbuild_dir, inplace, language_level):
        module_name = os.path.splitext(os.path.basename(filename))[0]
        super().__init__(module_name, filename)
        self._pyxbuild_dir = pyxbuild_dir
        self._inplace = inplace
        self._language_level = language_level

    def create_module(self, spec):
        try:
            so_path = build_module(spec.name, pyxfilename=spec.origin, pyxbuild_dir=self._pyxbuild_dir, inplace=self._inplace, language_level=self._language_level)
            self.path = so_path
            spec.origin = so_path
            return super().create_module(spec)
        except Exception as failure_exc:
            _debug('Failed to load extension module: %r' % failure_exc)
            if pyxargs.load_py_module_on_import_failure and spec.origin.endswith(PY_EXT):
                spec = importlib.util.spec_from_file_location(spec.name, spec.origin, loader=SourceFileLoader(spec.name, spec.origin))
                mod = importlib.util.module_from_spec(spec)
                assert mod.__file__ in (spec.origin, spec.origin + 'c', spec.origin + 'o'), (mod.__file__, spec.origin)
                return mod
            else:
                tb = sys.exc_info()[2]
                import traceback
                exc = ImportError('Building module %s failed: %s' % (spec.name, traceback.format_exception_only(*sys.exc_info()[:2])))
                raise exc.with_traceback(tb)

    def exec_module(self, module):
        try:
            return super().exec_module(module)
        except Exception as failure_exc:
            import traceback
            _debug('Failed to load extension module: %r' % failure_exc)
            raise ImportError('Executing module %s failed %s' % (module.__file__, traceback.format_exception_only(*sys.exc_info()[:2])))