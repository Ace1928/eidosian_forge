from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class AuthChallengeResponse:
    """
    Response to an AuthChallenge.
    """
    response: str
    username: typing.Optional[str] = None
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(response=str(json['response']), username=str(json['username']) if 'username' in json else None, password=str(json['password']) if 'password' in json else None)