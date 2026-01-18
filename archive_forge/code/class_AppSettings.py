import os
import functools
from enum import Enum
from pathlib import Path
from lazyops.types.models import BaseSettings, pre_root_validator, validator
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.utils.system import is_in_kubernetes, get_host_name
from lazyops.utils.assets import create_get_assets_wrapper, create_import_assets_wrapper
from lazyops.libs.fastapi_utils import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
class AppSettings(BaseAppSettings):
    """
    Custom App Settings
    """
    app_env: Optional[AppEnv] = None

    @validator('app_env', pre=True)
    def validate_app_env(cls, value: Optional[Any]) -> Any:
        """
        Validates the app environment
        """
        if value is None:
            return get_app_env(cls.__module__.split('.')[0])
        return AppEnv.from_env(value) if isinstance(value, str) else value

    def get_assets(self, *path_parts, load_file: Optional[bool]=False, loader: Optional[Callable]=None, **kwargs) -> Union[Path, Any, List[Path], List[Any], Dict[str, Path], Dict[str, Any]]:
        """
        Retrieves the assets

        args:
            path_parts: path parts to the assets directory (default: [])
            load_file: load the file (default: False)
            loader: loader function to use (default: None)
            **kwargs: additional arguments to pass to `get_module_assets`
        """
        get_assets = get_assets_func(self.module_name, 'assets')
        return get_assets(*path_parts, load_file=load_file, loader=loader, **kwargs)

    def import_assets(self, *path_parts, model: Optional[Type['BaseModel']]=None, load_file: Optional[bool]=False, **kwargs) -> Union[Dict[str, Any], List[Any]]:
        """
        Import assets from a module.

        args:
            path_parts: path parts to the assets directory (default: [])
            model: model to parse the assets with (default: None)
            load_file: load the file (default: False)
            **kwargs: additional arguments to pass to import_module_assets
        """
        import_assets = import_assets_func(self.module_name, 'assets')
        return import_assets(*path_parts, model=model, load_file=load_file, **kwargs)

    @property
    def is_local_env(self) -> bool:
        """
        Returns whether the environment is development
        """
        return self.app_env in [AppEnv.DEVELOPMENT, AppEnv.LOCAL] and (not self.in_k8s)

    @property
    def is_production_env(self) -> bool:
        """
        Returns whether the environment is production
        """
        return self.app_env == AppEnv.PRODUCTION and self.in_k8s

    @property
    def is_development_env(self) -> bool:
        """
        Returns whether the environment is development
        """
        return self.app_env in [AppEnv.DEVELOPMENT, AppEnv.LOCAL, AppEnv.CICD]