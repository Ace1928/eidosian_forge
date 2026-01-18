from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('FedCm.dialogShown')
@dataclass
class DialogShown:
    dialog_id: str
    dialog_type: DialogType
    accounts: typing.List[Account]
    title: str
    subtitle: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogShown:
        return cls(dialog_id=str(json['dialogId']), dialog_type=DialogType.from_json(json['dialogType']), accounts=[Account.from_json(i) for i in json['accounts']], title=str(json['title']), subtitle=str(json['subtitle']) if 'subtitle' in json else None)