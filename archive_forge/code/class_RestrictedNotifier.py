import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
class RestrictedNotifier(Notifier):
    """A notification class that restricts events registered/triggered.

    NOTE(harlowja): This class unlike :class:`.Notifier` restricts and
    disallows registering callbacks for event types that are not declared
    when constructing the notifier.
    """

    def __init__(self, watchable_events, allow_any=True):
        super(RestrictedNotifier, self).__init__()
        self._watchable_events = frozenset(watchable_events)
        self._allow_any = allow_any

    def events_iter(self):
        """Returns iterator of events that can be registered/subscribed to.

        NOTE(harlowja): does not include back the ``ANY`` event type as that
        meta-type is not a specific event but is a capture-all that does not
        imply the same meaning as specific event types.
        """
        for event_type in self._watchable_events:
            yield event_type

    def can_be_registered(self, event_type):
        """Checks if the event can be registered/subscribed to.

        :param event_type: event that needs to be verified
        :returns: whether the event can be registered/subscribed to
        :rtype: boolean
        """
        return event_type in self._watchable_events or (event_type == self.ANY and self._allow_any)