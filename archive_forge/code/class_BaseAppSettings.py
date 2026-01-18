import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
class BaseAppSettings(BaseSettings):
    """
    BaseSettings with additional helpers
    """

    @property
    def module_path(self) -> Path:
        """
        Gets the module root path

        https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
        """
        p = Path(pkg_resources.get_distribution(self.module_name).location)
        if 'src' in p.name and p.joinpath(self.module_name).exists():
            p = p.joinpath(self.module_name)
        elif p.joinpath('src').exists() and p.joinpath('src', self.module_name).exists():
            p = p.joinpath('src', self.module_name)
        return p

    @property
    def module_config_path(self) -> Path:
        """
        Returns the config module path
        """
        return Path(inspect.getfile(self.__class__)).parent

    @property
    def module_name(self) -> str:
        """
        Returns the module name
        """
        return self.__class__.__module__.split('.')[0]

    @property
    def module_version(self) -> str:
        """
        Returns the module version
        """
        return pkg_resources.get_distribution(self.module_name).version

    @property
    def module_pkg_name(self) -> str:
        """
        Returns the module pkg name
        
        {pkg}/src   -> src
        {pkg}/{pkg} -> {pkg}
        """
        config_path = self.module_config_path.as_posix()
        module_path = self.module_path.as_posix()
        return config_path.replace(module_path, '').strip().split('/', 2)[1]

    @property
    def in_k8s(self) -> bool:
        """
        Returns whether the app is running in kubernetes
        """
        from lazyops.utils.system import is_in_kubernetes
        return is_in_kubernetes()

    @property
    def host_name(self) -> str:
        """
        Returns the hostname
        """
        from lazyops.utils.system import get_host_name
        return get_host_name()