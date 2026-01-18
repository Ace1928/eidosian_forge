import concurrent.futures
from datetime import datetime
import enum
import logging
import json
import os
from typing import Optional
import aiohttp.web
from ray.dashboard.consts import RAY_CLUSTER_ACTIVITY_HOOK
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.storage import _load_class
from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc
from ray.dashboard.modules.job.common import JobInfoStorageClient
from ray._private.pydantic_compat import BaseModel, Extra, Field, validator
class RayActivityStatus(str, enum.Enum):
    ACTIVE = 'ACTIVE'
    INACTIVE = 'INACTIVE'
    ERROR = 'ERROR'