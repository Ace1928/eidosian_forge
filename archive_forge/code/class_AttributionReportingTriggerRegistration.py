from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingTriggerRegistration:
    filters: AttributionReportingFilterPair
    aggregatable_dedup_keys: typing.List[AttributionReportingAggregatableDedupKey]
    event_trigger_data: typing.List[AttributionReportingEventTriggerData]
    aggregatable_trigger_data: typing.List[AttributionReportingAggregatableTriggerData]
    aggregatable_values: typing.List[AttributionReportingAggregatableValueEntry]
    debug_reporting: bool
    source_registration_time_config: AttributionReportingSourceRegistrationTimeConfig
    debug_key: typing.Optional[UnsignedInt64AsBase10] = None
    aggregation_coordinator_origin: typing.Optional[str] = None
    trigger_context_id: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        json['aggregatableDedupKeys'] = [i.to_json() for i in self.aggregatable_dedup_keys]
        json['eventTriggerData'] = [i.to_json() for i in self.event_trigger_data]
        json['aggregatableTriggerData'] = [i.to_json() for i in self.aggregatable_trigger_data]
        json['aggregatableValues'] = [i.to_json() for i in self.aggregatable_values]
        json['debugReporting'] = self.debug_reporting
        json['sourceRegistrationTimeConfig'] = self.source_registration_time_config.to_json()
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        if self.trigger_context_id is not None:
            json['triggerContextId'] = self.trigger_context_id
        return json

    @classmethod
    def from_json(cls, json):
        return cls(filters=AttributionReportingFilterPair.from_json(json['filters']), aggregatable_dedup_keys=[AttributionReportingAggregatableDedupKey.from_json(i) for i in json['aggregatableDedupKeys']], event_trigger_data=[AttributionReportingEventTriggerData.from_json(i) for i in json['eventTriggerData']], aggregatable_trigger_data=[AttributionReportingAggregatableTriggerData.from_json(i) for i in json['aggregatableTriggerData']], aggregatable_values=[AttributionReportingAggregatableValueEntry.from_json(i) for i in json['aggregatableValues']], debug_reporting=bool(json['debugReporting']), source_registration_time_config=AttributionReportingSourceRegistrationTimeConfig.from_json(json['sourceRegistrationTimeConfig']), debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None, aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None, trigger_context_id=str(json['triggerContextId']) if 'triggerContextId' in json else None)