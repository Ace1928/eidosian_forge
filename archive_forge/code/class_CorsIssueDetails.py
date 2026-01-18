from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class CorsIssueDetails:
    """
    Details for a CORS related issue, e.g. a warning or error related to
    CORS RFC1918 enforcement.
    """
    cors_error_status: network.CorsErrorStatus
    is_warning: bool
    request: AffectedRequest
    location: typing.Optional[SourceCodeLocation] = None
    initiator_origin: typing.Optional[str] = None
    resource_ip_address_space: typing.Optional[network.IPAddressSpace] = None
    client_security_state: typing.Optional[network.ClientSecurityState] = None

    def to_json(self):
        json = dict()
        json['corsErrorStatus'] = self.cors_error_status.to_json()
        json['isWarning'] = self.is_warning
        json['request'] = self.request.to_json()
        if self.location is not None:
            json['location'] = self.location.to_json()
        if self.initiator_origin is not None:
            json['initiatorOrigin'] = self.initiator_origin
        if self.resource_ip_address_space is not None:
            json['resourceIPAddressSpace'] = self.resource_ip_address_space.to_json()
        if self.client_security_state is not None:
            json['clientSecurityState'] = self.client_security_state.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(cors_error_status=network.CorsErrorStatus.from_json(json['corsErrorStatus']), is_warning=bool(json['isWarning']), request=AffectedRequest.from_json(json['request']), location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None, initiator_origin=str(json['initiatorOrigin']) if 'initiatorOrigin' in json else None, resource_ip_address_space=network.IPAddressSpace.from_json(json['resourceIPAddressSpace']) if 'resourceIPAddressSpace' in json else None, client_security_state=network.ClientSecurityState.from_json(json['clientSecurityState']) if 'clientSecurityState' in json else None)