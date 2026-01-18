import _imp
import _io
import sys
import _warnings
import marshal
class WindowsRegistryFinder:
    """Meta path finder for modules declared in the Windows registry."""
    REGISTRY_KEY = 'Software\\Python\\PythonCore\\{sys_version}\\Modules\\{fullname}'
    REGISTRY_KEY_DEBUG = 'Software\\Python\\PythonCore\\{sys_version}\\Modules\\{fullname}\\Debug'
    DEBUG_BUILD = _MS_WINDOWS and '_d.pyd' in EXTENSION_SUFFIXES

    @staticmethod
    def _open_registry(key):
        try:
            return winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
        except OSError:
            return winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)

    @classmethod
    def _search_registry(cls, fullname):
        if cls.DEBUG_BUILD:
            registry_key = cls.REGISTRY_KEY_DEBUG
        else:
            registry_key = cls.REGISTRY_KEY
        key = registry_key.format(fullname=fullname, sys_version='%d.%d' % sys.version_info[:2])
        try:
            with cls._open_registry(key) as hkey:
                filepath = winreg.QueryValue(hkey, '')
        except OSError:
            return None
        return filepath

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        filepath = cls._search_registry(fullname)
        if filepath is None:
            return None
        try:
            _path_stat(filepath)
        except OSError:
            return None
        for loader, suffixes in _get_supported_file_loaders():
            if filepath.endswith(tuple(suffixes)):
                spec = _bootstrap.spec_from_loader(fullname, loader(fullname, filepath), origin=filepath)
                return spec

    @classmethod
    def find_module(cls, fullname, path=None):
        """Find module named in the registry.

        This method is deprecated.  Use find_spec() instead.

        """
        _warnings.warn('WindowsRegistryFinder.find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        spec = cls.find_spec(fullname, path)
        if spec is not None:
            return spec.loader
        else:
            return None