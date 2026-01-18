from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
@dataclass
class PermissionDescriptor:
    """
    Definition of PermissionDescriptor defined in the Permissions API:
    https://w3c.github.io/permissions/#dom-permissiondescriptor.
    """
    name: str
    sysex: typing.Optional[bool] = None
    user_visible_only: typing.Optional[bool] = None
    allow_without_sanitization: typing.Optional[bool] = None
    pan_tilt_zoom: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.sysex is not None:
            json['sysex'] = self.sysex
        if self.user_visible_only is not None:
            json['userVisibleOnly'] = self.user_visible_only
        if self.allow_without_sanitization is not None:
            json['allowWithoutSanitization'] = self.allow_without_sanitization
        if self.pan_tilt_zoom is not None:
            json['panTiltZoom'] = self.pan_tilt_zoom
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), sysex=bool(json['sysex']) if 'sysex' in json else None, user_visible_only=bool(json['userVisibleOnly']) if 'userVisibleOnly' in json else None, allow_without_sanitization=bool(json['allowWithoutSanitization']) if 'allowWithoutSanitization' in json else None, pan_tilt_zoom=bool(json['panTiltZoom']) if 'panTiltZoom' in json else None)