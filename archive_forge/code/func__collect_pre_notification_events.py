import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
def _collect_pre_notification_events(self, *args):
    self.pre_change_events.append(args)