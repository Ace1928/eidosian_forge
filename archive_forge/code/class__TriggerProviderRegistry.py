from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
class _TriggerProviderRegistry(object):
    """This class encapsulates all Event Trigger related functionality."""

    def __init__(self, all_providers):
        self.providers = all_providers
        self._unadvertised_provider = TriggerProvider(UNADVERTISED_PROVIDER_LABEL, [])

    def ProvidersLabels(self):
        return (p.label for p in self.providers)

    def Provider(self, provider):
        return next((p for p in self.providers if p.label == provider))

    def EventsLabels(self, provider):
        return (e.label for e in self.Provider(provider).events)

    def AllEventLabels(self):
        all_events = (self.EventsLabels(p.label) for p in self.providers)
        return itertools.chain.from_iterable(all_events)

    def Event(self, provider, event):
        return next((e for e in self.Provider(provider).events if e.label == event))

    def ProviderForEvent(self, event_label):
        for p in self.providers:
            if event_label in self.EventsLabels(p.label):
                return p
        return self._unadvertised_provider