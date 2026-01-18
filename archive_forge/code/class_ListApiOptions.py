import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
@dataclass(init=not IS_PYDANTIC_2)
class ListApiOptions:
    limit: int = DEFAULT_LIMIT
    timeout: int = DEFAULT_RPC_TIMEOUT
    detail: bool = False
    filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]] = field(default_factory=list)
    exclude_driver: bool = True
    server_timeout_multiplier: float = 0.8

    def __post_init__(self):
        self.timeout = int(self.timeout * self.server_timeout_multiplier)
        assert self.timeout != 0, '0 second timeout is not supported.'
        if self.filters is None:
            self.filters = []
        for filter in self.filters:
            _, filter_predicate, _ = filter
            if filter_predicate != '=' and filter_predicate != '!=':
                raise ValueError(f'Unsupported filter predicate {filter_predicate} is given. Available predicates: =, !=.')