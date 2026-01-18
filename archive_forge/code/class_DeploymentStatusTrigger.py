import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, NamedTuple, Optional
from ray.actor import ActorHandle
from ray.serve.generated.serve_pb2 import ApplicationStatus as ApplicationStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import DeploymentStatus as DeploymentStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
class DeploymentStatusTrigger(str, Enum):
    UNSPECIFIED = 'UNSPECIFIED'
    CONFIG_UPDATE_STARTED = 'CONFIG_UPDATE_STARTED'
    CONFIG_UPDATE_COMPLETED = 'CONFIG_UPDATE_COMPLETED'
    UPSCALE_COMPLETED = 'UPSCALE_COMPLETED'
    DOWNSCALE_COMPLETED = 'DOWNSCALE_COMPLETED'
    AUTOSCALING = 'AUTOSCALING'
    REPLICA_STARTUP_FAILED = 'REPLICA_STARTUP_FAILED'
    HEALTH_CHECK_FAILED = 'HEALTH_CHECK_FAILED'
    INTERNAL_ERROR = 'INTERNAL_ERROR'
    DELETING = 'DELETING'