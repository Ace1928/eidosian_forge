from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@dataclass
class ServiceWorkerVersion:
    """
    ServiceWorker version.
    """
    version_id: str
    registration_id: RegistrationID
    script_url: str
    running_status: ServiceWorkerVersionRunningStatus
    status: ServiceWorkerVersionStatus
    script_last_modified: typing.Optional[float] = None
    script_response_time: typing.Optional[float] = None
    controlled_clients: typing.Optional[typing.List[target.TargetID]] = None
    target_id: typing.Optional[target.TargetID] = None

    def to_json(self):
        json = dict()
        json['versionId'] = self.version_id
        json['registrationId'] = self.registration_id.to_json()
        json['scriptURL'] = self.script_url
        json['runningStatus'] = self.running_status.to_json()
        json['status'] = self.status.to_json()
        if self.script_last_modified is not None:
            json['scriptLastModified'] = self.script_last_modified
        if self.script_response_time is not None:
            json['scriptResponseTime'] = self.script_response_time
        if self.controlled_clients is not None:
            json['controlledClients'] = [i.to_json() for i in self.controlled_clients]
        if self.target_id is not None:
            json['targetId'] = self.target_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(version_id=str(json['versionId']), registration_id=RegistrationID.from_json(json['registrationId']), script_url=str(json['scriptURL']), running_status=ServiceWorkerVersionRunningStatus.from_json(json['runningStatus']), status=ServiceWorkerVersionStatus.from_json(json['status']), script_last_modified=float(json['scriptLastModified']) if 'scriptLastModified' in json else None, script_response_time=float(json['scriptResponseTime']) if 'scriptResponseTime' in json else None, controlled_clients=[target.TargetID.from_json(i) for i in json['controlledClients']] if 'controlledClients' in json else None, target_id=target.TargetID.from_json(json['targetId']) if 'targetId' in json else None)