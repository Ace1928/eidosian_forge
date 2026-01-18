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
class DeploymentID(NamedTuple):
    name: str
    app: str

    def __str__(self):
        if self.app:
            return f'{self.app}_{self.name}'
        else:
            return self.name

    def to_replica_actor_class_name(self):
        if self.app:
            return f'ServeReplica:{self.app}:{self.name}'
        else:
            return f'ServeReplica:{self.name}'