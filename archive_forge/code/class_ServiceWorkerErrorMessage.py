from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@dataclass
class ServiceWorkerErrorMessage:
    """
    ServiceWorker error message.
    """
    error_message: str
    registration_id: RegistrationID
    version_id: str
    source_url: str
    line_number: int
    column_number: int

    def to_json(self):
        json = dict()
        json['errorMessage'] = self.error_message
        json['registrationId'] = self.registration_id.to_json()
        json['versionId'] = self.version_id
        json['sourceURL'] = self.source_url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(error_message=str(json['errorMessage']), registration_id=RegistrationID.from_json(json['registrationId']), version_id=str(json['versionId']), source_url=str(json['sourceURL']), line_number=int(json['lineNumber']), column_number=int(json['columnNumber']))