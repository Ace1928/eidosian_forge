import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def _validate_runtime_env(self):
    from ray.runtime_env import RuntimeEnv
    if isinstance(self.runtime_env, RuntimeEnv):
        return self.runtime_env
    return RuntimeEnv(**self.runtime_env)