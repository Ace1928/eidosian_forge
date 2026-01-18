from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class LoginState(enum.Enum):
    """
    Whether this is a sign-up or sign-in action for this account, i.e.
    whether this account has ever been used to sign in to this RP before.
    """
    SIGN_IN = 'SignIn'
    SIGN_UP = 'SignUp'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)