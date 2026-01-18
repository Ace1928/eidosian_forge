import unittest
from traits.api import (
from traits.observation.api import (
class Album(HasTraits):
    records = List(Instance(Record))
    records_default_call_count = Int()
    record_number_change_events = List()
    name_to_records = Dict(Str, Record)
    name_to_records_default_call_count = Int()
    name_to_records_clicked_events = List()

    def _records_default(self):
        self.records_default_call_count += 1
        return [Record()]

    @observe(trait('records').list_items().trait('number'))
    def handle_record_number_changed(self, event):
        self.record_number_change_events.append(event)

    def _name_to_records_default(self):
        self.name_to_records_default_call_count += 1
        return {'Record': Record()}

    @observe('name_to_records:items:clicked')
    def handle_event(self, event):
        self.name_to_records_clicked_events.append(event)