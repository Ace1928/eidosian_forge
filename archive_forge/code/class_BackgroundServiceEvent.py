from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
@dataclass
class BackgroundServiceEvent:
    timestamp: network.TimeSinceEpoch
    origin: str
    service_worker_registration_id: service_worker.RegistrationID
    service: ServiceName
    event_name: str
    instance_id: str
    event_metadata: typing.List[EventMetadata]

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['origin'] = self.origin
        json['serviceWorkerRegistrationId'] = self.service_worker_registration_id.to_json()
        json['service'] = self.service.to_json()
        json['eventName'] = self.event_name
        json['instanceId'] = self.instance_id
        json['eventMetadata'] = [i.to_json() for i in self.event_metadata]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(timestamp=network.TimeSinceEpoch.from_json(json['timestamp']), origin=str(json['origin']), service_worker_registration_id=service_worker.RegistrationID.from_json(json['serviceWorkerRegistrationId']), service=ServiceName.from_json(json['service']), event_name=str(json['eventName']), instance_id=str(json['instanceId']), event_metadata=[EventMetadata.from_json(i) for i in json['eventMetadata']])