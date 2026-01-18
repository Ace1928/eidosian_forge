import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def _get_serialized_runtime_env(self) -> str:
    """Return the JSON-serialized parsed runtime env dict"""
    return self._validate_runtime_env().serialize()