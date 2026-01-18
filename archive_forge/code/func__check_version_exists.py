import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
import numpy as np
from gym.wrappers import (
from gym.wrappers.compatibility import EnvCompatibility
from gym.wrappers.env_checker import PassiveEnvChecker
from gym import Env, error, logger
def _check_version_exists(ns: Optional[str], name: str, version: Optional[int]):
    """Check if an env version exists in a namespace. If it doesn't, print a helpful error message.
    This is a complete test whether an environment identifier is valid, and will provide the best available hints.

    Args:
        ns: The environment namespace
        name: The environment space
        version: The environment version

    Raises:
        DeprecatedEnv: The environment doesn't exist but a default version does
        VersionNotFound: The ``version`` used doesn't exist
        DeprecatedEnv: Environment version is deprecated
    """
    if get_env_id(ns, name, version) in registry:
        return
    _check_name_exists(ns, name)
    if version is None:
        return
    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` doesn't exist."
    env_specs = [spec_ for spec_ in registry.values() if spec_.namespace == ns and spec_.name == name]
    env_specs = sorted(env_specs, key=lambda spec_: int(spec_.version or -1))
    default_spec = [spec_ for spec_ in env_specs if spec_.version is None]
    if default_spec:
        message += f' It provides the default version {default_spec[0].id}`.'
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)
    versioned_specs = [spec_ for spec_ in env_specs if spec_.version is not None]
    latest_spec = max(versioned_specs, key=lambda spec: spec.version, default=None)
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ', '.join((f'`v{spec_.version}`' for spec_ in env_specs))
        message += f' It provides versioned environments: [ {version_list_msg} ].'
        raise error.VersionNotFound(message)
    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(f'Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. Please use `{latest_spec.id}` instead.')