import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import ray
from ray._private.ray_constants import DEFAULT_RUNTIME_ENV_TIMEOUT_SECONDS
from ray._private.runtime_env.conda import get_uri as get_conda_uri
from ray._private.runtime_env.pip import get_uri as get_pip_uri
from ray._private.runtime_env.plugin_schema_manager import RuntimeEnvPluginSchemaManager
from ray._private.runtime_env.validation import OPTION_TO_VALIDATION_FN
from ray._private.thirdparty.dacite import from_dict
from ray.core.generated.runtime_env_common_pb2 import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
class RuntimeEnvConfig(dict):
    """Used to specify configuration options for a runtime environment.

    The config is not included when calculating the runtime_env hash,
    which means that two runtime_envs with the same options but different
    configs are considered the same for caching purposes.

    Args:
        setup_timeout_seconds: The timeout of runtime environment
            creation, timeout is in seconds. The value `-1` means disable
            timeout logic, except `-1`, `setup_timeout_seconds` cannot be
            less than or equal to 0. The default value of `setup_timeout_seconds`
            is 600 seconds.
        eager_install: Indicates whether to install the runtime environment
            on the cluster at `ray.init()` time, before the workers are leased.
            This flag is set to `True` by default.
    """
    known_fields: Set[str] = {'setup_timeout_seconds', 'eager_install'}
    _default_config: Dict = {'setup_timeout_seconds': DEFAULT_RUNTIME_ENV_TIMEOUT_SECONDS, 'eager_install': True}

    def __init__(self, setup_timeout_seconds: int=DEFAULT_RUNTIME_ENV_TIMEOUT_SECONDS, eager_install: bool=True):
        super().__init__()
        if not isinstance(setup_timeout_seconds, int):
            raise TypeError(f'setup_timeout_seconds must be of type int, got: {type(setup_timeout_seconds)}')
        elif setup_timeout_seconds <= 0 and setup_timeout_seconds != -1:
            raise ValueError(f'setup_timeout_seconds must be greater than zero or equals to -1, got: {setup_timeout_seconds}')
        self['setup_timeout_seconds'] = setup_timeout_seconds
        if not isinstance(eager_install, bool):
            raise TypeError(f'eager_install must be a boolean. got {type(eager_install)}')
        self['eager_install'] = eager_install

    @staticmethod
    def parse_and_validate_runtime_env_config(config: Union[Dict, 'RuntimeEnvConfig']) -> 'RuntimeEnvConfig':
        if isinstance(config, RuntimeEnvConfig):
            return config
        elif isinstance(config, Dict):
            unknown_fields = set(config.keys()) - RuntimeEnvConfig.known_fields
            if len(unknown_fields):
                logger.warning(f'The following unknown entries in the runtime_env_config dictionary will be ignored: {unknown_fields}.')
            config_dict = dict()
            for field in RuntimeEnvConfig.known_fields:
                if field in config:
                    config_dict[field] = config[field]
            return RuntimeEnvConfig(**config_dict)
        else:
            raise TypeError(f"runtime_env['config'] must be of type dict or RuntimeEnvConfig, got: {type(config)}")

    @classmethod
    def default_config(cls):
        return RuntimeEnvConfig(**cls._default_config)

    def build_proto_runtime_env_config(self) -> ProtoRuntimeEnvConfig:
        runtime_env_config = ProtoRuntimeEnvConfig()
        runtime_env_config.setup_timeout_seconds = self['setup_timeout_seconds']
        runtime_env_config.eager_install = self['eager_install']
        return runtime_env_config

    @classmethod
    def from_proto(cls, runtime_env_config: ProtoRuntimeEnvConfig):
        setup_timeout_seconds = runtime_env_config.setup_timeout_seconds
        if setup_timeout_seconds == 0:
            setup_timeout_seconds = cls._default_config['setup_timeout_seconds']
        return cls(setup_timeout_seconds=setup_timeout_seconds, eager_install=runtime_env_config.eager_install)

    def to_dict(self) -> Dict:
        return dict(deepcopy(self))