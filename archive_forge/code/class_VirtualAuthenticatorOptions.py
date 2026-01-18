from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class VirtualAuthenticatorOptions:
    protocol: AuthenticatorProtocol
    transport: AuthenticatorTransport
    has_resident_key: typing.Optional[bool] = None
    has_user_verification: typing.Optional[bool] = None
    automatic_presence_simulation: typing.Optional[bool] = None
    is_user_verified: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol.to_json()
        json['transport'] = self.transport.to_json()
        if self.has_resident_key is not None:
            json['hasResidentKey'] = self.has_resident_key
        if self.has_user_verification is not None:
            json['hasUserVerification'] = self.has_user_verification
        if self.automatic_presence_simulation is not None:
            json['automaticPresenceSimulation'] = self.automatic_presence_simulation
        if self.is_user_verified is not None:
            json['isUserVerified'] = self.is_user_verified
        return json

    @classmethod
    def from_json(cls, json):
        return cls(protocol=AuthenticatorProtocol.from_json(json['protocol']), transport=AuthenticatorTransport.from_json(json['transport']), has_resident_key=bool(json['hasResidentKey']) if 'hasResidentKey' in json else None, has_user_verification=bool(json['hasUserVerification']) if 'hasUserVerification' in json else None, automatic_presence_simulation=bool(json['automaticPresenceSimulation']) if 'automaticPresenceSimulation' in json else None, is_user_verified=bool(json['isUserVerified']) if 'isUserVerified' in json else None)