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
class ApplicationStatus(str, Enum):
    NOT_STARTED = 'NOT_STARTED'
    DEPLOYING = 'DEPLOYING'
    DEPLOY_FAILED = 'DEPLOY_FAILED'
    RUNNING = 'RUNNING'
    UNHEALTHY = 'UNHEALTHY'
    DELETING = 'DELETING'