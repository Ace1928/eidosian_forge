from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
@dataclass
class TraceConfig:
    record_mode: typing.Optional[str] = None
    enable_sampling: typing.Optional[bool] = None
    enable_systrace: typing.Optional[bool] = None
    enable_argument_filter: typing.Optional[bool] = None
    included_categories: typing.Optional[typing.List[str]] = None
    excluded_categories: typing.Optional[typing.List[str]] = None
    synthetic_delays: typing.Optional[typing.List[str]] = None
    memory_dump_config: typing.Optional[MemoryDumpConfig] = None

    def to_json(self):
        json = dict()
        if self.record_mode is not None:
            json['recordMode'] = self.record_mode
        if self.enable_sampling is not None:
            json['enableSampling'] = self.enable_sampling
        if self.enable_systrace is not None:
            json['enableSystrace'] = self.enable_systrace
        if self.enable_argument_filter is not None:
            json['enableArgumentFilter'] = self.enable_argument_filter
        if self.included_categories is not None:
            json['includedCategories'] = [i for i in self.included_categories]
        if self.excluded_categories is not None:
            json['excludedCategories'] = [i for i in self.excluded_categories]
        if self.synthetic_delays is not None:
            json['syntheticDelays'] = [i for i in self.synthetic_delays]
        if self.memory_dump_config is not None:
            json['memoryDumpConfig'] = self.memory_dump_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(record_mode=str(json['recordMode']) if 'recordMode' in json else None, enable_sampling=bool(json['enableSampling']) if 'enableSampling' in json else None, enable_systrace=bool(json['enableSystrace']) if 'enableSystrace' in json else None, enable_argument_filter=bool(json['enableArgumentFilter']) if 'enableArgumentFilter' in json else None, included_categories=[str(i) for i in json['includedCategories']] if 'includedCategories' in json else None, excluded_categories=[str(i) for i in json['excludedCategories']] if 'excludedCategories' in json else None, synthetic_delays=[str(i) for i in json['syntheticDelays']] if 'syntheticDelays' in json else None, memory_dump_config=MemoryDumpConfig.from_json(json['memoryDumpConfig']) if 'memoryDumpConfig' in json else None)