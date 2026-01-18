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
class AppEnv(str, Enum):
    CICD = 'cicd'
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'
    LOCAL = 'local'

    @classmethod
    def from_env(cls, env_value: str) -> 'AppEnv':
        """
        Get the app environment from the environment variables
        """
        env_value = env_value.lower()
        if 'cicd' in env_value or 'ci/cd' in env_value:
            return cls.CICD
        if 'prod' in env_value:
            return cls.PRODUCTION
        if 'dev' in env_value:
            return cls.DEVELOPMENT
        if 'staging' in env_value:
            return cls.STAGING
        if 'local' in env_value:
            return cls.LOCAL
        raise ValueError(f'Invalid app environment: {env_value} ({type(env_value)})')

    def __eq__(self, other: Any) -> bool:
        """
        Equality operator
        """
        if isinstance(other, str):
            return self.value == other.lower()
        return self.value == other.value if isinstance(other, AppEnv) else False

    @property
    def is_devel(self) -> bool:
        """
        Returns True if the app environment is development
        """
        return self in [AppEnv.LOCAL, AppEnv.CICD, AppEnv.DEVELOPMENT]

    @property
    def is_local(self) -> bool:
        """
        Returns True if the app environment is local
        """
        return self in [AppEnv.LOCAL, AppEnv.CICD]

    @property
    def name(self) -> str:
        """
        Returns the name in lower
        """
        return self.value.lower()