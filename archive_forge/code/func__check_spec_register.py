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
def _check_spec_register(spec: EnvSpec):
    """Checks whether the spec is valid to be registered. Helper function for `register`."""
    global registry
    latest_versioned_spec = max((spec_ for spec_ in registry.values() if spec_.namespace == spec.namespace and spec_.name == spec.name and (spec_.version is not None)), key=lambda spec_: int(spec_.version), default=None)
    unversioned_spec = next((spec_ for spec_ in registry.values() if spec_.namespace == spec.namespace and spec_.name == spec.name and (spec_.version is None)), None)
    if unversioned_spec is not None and spec.version is not None:
        raise error.RegistrationError(f"Can't register the versioned environment `{spec.id}` when the unversioned environment `{unversioned_spec.id}` of the same name already exists.")
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.RegistrationError(f"Can't register the unversioned environment `{spec.id}` when the versioned environment `{latest_versioned_spec.id}` of the same name already exists. Note: the default behavior is that `gym.make` with the unversioned environment will return the latest versioned environment")