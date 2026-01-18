from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAuthn.credentialAdded')
@dataclass
class CredentialAdded:
    """
    Triggered when a credential is added to an authenticator.
    """
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAdded:
        return cls(authenticator_id=AuthenticatorId.from_json(json['authenticatorId']), credential=Credential.from_json(json['credential']))