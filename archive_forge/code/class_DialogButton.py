from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class DialogButton(enum.Enum):
    """
    The buttons on the FedCM dialog.
    """
    CONFIRM_IDP_LOGIN_CONTINUE = 'ConfirmIdpLoginContinue'
    ERROR_GOT_IT = 'ErrorGotIt'
    ERROR_MORE_DETAILS = 'ErrorMoreDetails'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)