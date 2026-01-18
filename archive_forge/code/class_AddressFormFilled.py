from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@event_class('Autofill.addressFormFilled')
@dataclass
class AddressFormFilled:
    """
    Emitted when an address form is filled.
    """
    filled_fields: typing.List[FilledField]
    address_ui: AddressUI

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddressFormFilled:
        return cls(filled_fields=[FilledField.from_json(i) for i in json['filledFields']], address_ui=AddressUI.from_json(json['addressUi']))