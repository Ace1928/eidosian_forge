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
@dataclass(eq=True)
class DeploymentStatusInfo:
    name: str
    status: DeploymentStatus
    status_trigger: DeploymentStatusTrigger
    message: str = ''

    def debug_string(self):
        return json.dumps(asdict(self), indent=4)

    def update(self, status: DeploymentStatus=None, status_trigger: DeploymentStatusTrigger=None, message: str=''):
        return DeploymentStatusInfo(name=self.name, status=status if status else self.status, status_trigger=status_trigger if status_trigger else self.status_trigger, message=message)

    def to_proto(self):
        return DeploymentStatusInfoProto(name=self.name, status=f'DEPLOYMENT_STATUS_{self.status.name}', status_trigger=f'DEPLOYMENT_STATUS_TRIGGER_{self.status_trigger.name}', message=self.message)

    @classmethod
    def from_proto(cls, proto: DeploymentStatusInfoProto):
        status = DeploymentStatusProto.Name(proto.status)[len('DEPLOYMENT_STATUS_'):]
        status_trigger = DeploymentStatusTriggerProto.Name(proto.status_trigger)[len('DEPLOYMENT_STATUS_TRIGGER_'):]
        return cls(name=proto.name, status=DeploymentStatus(status), status_trigger=DeploymentStatusTrigger(status_trigger), message=proto.message)