from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class GPUInfo:
    """
    Provides information about the GPU(s) on the system.
    """
    devices: typing.List[GPUDevice]
    driver_bug_workarounds: typing.List[str]
    video_decoding: typing.List[VideoDecodeAcceleratorCapability]
    video_encoding: typing.List[VideoEncodeAcceleratorCapability]
    image_decoding: typing.List[ImageDecodeAcceleratorCapability]
    aux_attributes: typing.Optional[dict] = None
    feature_status: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['devices'] = [i.to_json() for i in self.devices]
        json['driverBugWorkarounds'] = [i for i in self.driver_bug_workarounds]
        json['videoDecoding'] = [i.to_json() for i in self.video_decoding]
        json['videoEncoding'] = [i.to_json() for i in self.video_encoding]
        json['imageDecoding'] = [i.to_json() for i in self.image_decoding]
        if self.aux_attributes is not None:
            json['auxAttributes'] = self.aux_attributes
        if self.feature_status is not None:
            json['featureStatus'] = self.feature_status
        return json

    @classmethod
    def from_json(cls, json):
        return cls(devices=[GPUDevice.from_json(i) for i in json['devices']], driver_bug_workarounds=[str(i) for i in json['driverBugWorkarounds']], video_decoding=[VideoDecodeAcceleratorCapability.from_json(i) for i in json['videoDecoding']], video_encoding=[VideoEncodeAcceleratorCapability.from_json(i) for i in json['videoEncoding']], image_decoding=[ImageDecodeAcceleratorCapability.from_json(i) for i in json['imageDecoding']], aux_attributes=dict(json['auxAttributes']) if 'auxAttributes' in json else None, feature_status=dict(json['featureStatus']) if 'featureStatus' in json else None)