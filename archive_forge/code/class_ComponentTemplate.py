from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ComponentTemplate:
    """Class that wraps a ComponentTemplate JSON object."""

    def __init__(self, name, event_input, description):
        self.name = name
        self.event_input = event_input
        self.description = description

    @classmethod
    def FromJSON(cls, json_object):
        return cls(name=json_object['name'], event_input=json_object['event-input'], description=json_object['description'])

    def __repr__(self):
        return '<ComponentTemplate: name="{0.name}" event_input="{0.event_input}" description="{0.description}">'.format(self)