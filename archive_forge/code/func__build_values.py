import os
import warnings
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from .config import BaseConfig, Extra
from .fields import ModelField
from .main import BaseModel
from .types import JsonWrapper
from .typing import StrPath, display_as_type, get_origin, is_union
from .utils import deep_update, lenient_issubclass, path_type, sequence_like
def _build_values(self, init_kwargs: Dict[str, Any], _env_file: Optional[DotenvType]=None, _env_file_encoding: Optional[str]=None, _env_nested_delimiter: Optional[str]=None, _secrets_dir: Optional[StrPath]=None) -> Dict[str, Any]:
    init_settings = InitSettingsSource(init_kwargs=init_kwargs)
    env_settings = EnvSettingsSource(env_file=_env_file if _env_file != env_file_sentinel else self.__config__.env_file, env_file_encoding=_env_file_encoding if _env_file_encoding is not None else self.__config__.env_file_encoding, env_nested_delimiter=_env_nested_delimiter if _env_nested_delimiter is not None else self.__config__.env_nested_delimiter, env_prefix_len=len(self.__config__.env_prefix))
    file_secret_settings = SecretsSettingsSource(secrets_dir=_secrets_dir or self.__config__.secrets_dir)
    sources = self.__config__.customise_sources(init_settings=init_settings, env_settings=env_settings, file_secret_settings=file_secret_settings)
    if sources:
        return deep_update(*reversed([source(self) for source in sources]))
    else:
        return {}