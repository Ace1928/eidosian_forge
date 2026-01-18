from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('DeviceAccess.deviceRequestPrompted')
@dataclass
class DeviceRequestPrompted:
    """
    A device request opened a user prompt to select a device. Respond with the
    selectPrompt or cancelPrompt command.
    """
    id_: RequestId
    devices: typing.List[PromptDevice]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DeviceRequestPrompted:
        return cls(id_=RequestId.from_json(json['id']), devices=[PromptDevice.from_json(i) for i in json['devices']])