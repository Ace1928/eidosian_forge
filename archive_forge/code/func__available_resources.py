from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
@property
def _available_resources(self) -> Dict[str, float]:
    available_resources = self._total_resources.copy()
    for used_resources in self._used_resources:
        all_resources = used_resources.required_resources
        for k, v in all_resources.items():
            available_resources[k] = (available_resources[k] * _DIGITS - v * _DIGITS) / _DIGITS
    return available_resources