import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
class Notifier(object):
    """A notification (`pub/sub`_ *like*) helper class.

    It is intended to be used to subscribe to notifications of events
    occurring as well as allow a entity to post said notifications to any
    associated subscribers without having either entity care about how this
    notification occurs.

    **Not** thread-safe when a single notifier is mutated at the same
    time by multiple threads. For example having multiple threads call
    into :py:meth:`.register` or :py:meth:`.reset` at the same time could
    potentially end badly. It is thread-safe when
    only :py:meth:`.notify` calls or other read-only actions (like calling
    into :py:meth:`.is_registered`) are occurring at the same time.

    .. _pub/sub: http://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern
    """
    RESERVED_KEYS = ('details',)
    ANY = '*'
    _DISALLOWED_NOTIFICATION_EVENTS = set([ANY])

    def __init__(self):
        self._topics = collections.defaultdict(list)

    def __len__(self):
        """Returns how many callbacks are registered.

        :returns: count of how many callbacks are registered
        :rtype: number
        """
        count = 0
        for _event_type, listeners in self._topics.items():
            count += len(listeners)
        return count

    def is_registered(self, event_type, callback, details_filter=None):
        """Check if a callback is registered.

        :returns: checks if the callback is registered
        :rtype: boolean
        """
        for listener in self._topics.get(event_type, []):
            if listener.is_equivalent(callback, details_filter=details_filter):
                return True
        return False

    def reset(self):
        """Forget all previously registered callbacks."""
        self._topics.clear()

    def notify(self, event_type, details):
        """Notify about event occurrence.

        All callbacks registered to receive notifications about given
        event type will be called. If the provided event type can not be
        used to emit notifications (this is checked via
        the :meth:`.can_be_registered` method) then it will silently be
        dropped (notification failures are not allowed to cause or
        raise exceptions).

        :param event_type: event type that occurred
        :param details: additional event details *dictionary* passed to
                        callback keyword argument with the same name
        :type details: dictionary
        """
        if not self.can_trigger_notification(event_type):
            LOG.debug("Event type '%s' is not allowed to trigger notifications", event_type)
            return
        listeners = list(self._topics.get(self.ANY, []))
        listeners.extend(self._topics.get(event_type, []))
        if not listeners:
            return
        if not details:
            details = {}
        for listener in listeners:
            try:
                listener(event_type, details.copy())
            except Exception:
                LOG.warning('Failure calling listener %s to notify about event %s, details: %s', listener, event_type, details, exc_info=True)

    def register(self, event_type, callback, args=None, kwargs=None, details_filter=None):
        """Register a callback to be called when event of a given type occurs.

        Callback will be called with provided ``args`` and ``kwargs`` and
        when event type occurs (or on any event if ``event_type`` equals to
        :attr:`.ANY`). It will also get additional keyword argument,
        ``details``, that will hold event details provided to the
        :meth:`.notify` method (if a details filter callback is provided then
        the target callback will *only* be triggered if the details filter
        callback returns a truthy value).

        :param event_type: event type input
        :param callback: function callback to be registered.
        :param args: non-keyworded arguments
        :type args: list
        :param kwargs: key-value pair arguments
        :type kwargs: dictionary
        """
        if not callable(callback):
            raise ValueError('Event callback must be callable')
        if details_filter is not None:
            if not callable(details_filter):
                raise ValueError('Details filter must be callable')
        if not self.can_be_registered(event_type):
            raise ValueError("Disallowed event type '%s' can not have a callback registered" % event_type)
        if self.is_registered(event_type, callback, details_filter=details_filter):
            raise ValueError('Event callback already registered with equivalent details filter')
        if kwargs:
            for k in self.RESERVED_KEYS:
                if k in kwargs:
                    raise KeyError("Reserved key '%s' not allowed in kwargs" % k)
        self._topics[event_type].append(Listener(callback, args=args, kwargs=kwargs, details_filter=details_filter))

    def deregister(self, event_type, callback, details_filter=None):
        """Remove a single listener bound to event ``event_type``.

        :param event_type: deregister listener bound to event_type
        """
        if event_type not in self._topics:
            return False
        for i, listener in enumerate(self._topics.get(event_type, [])):
            if listener.is_equivalent(callback, details_filter=details_filter):
                self._topics[event_type].pop(i)
                return True
        return False

    def deregister_event(self, event_type):
        """Remove a group of listeners bound to event ``event_type``.

        :param event_type: deregister listeners bound to event_type
        """
        return len(self._topics.pop(event_type, []))

    def copy(self):
        c = copy.copy(self)
        c._topics = collections.defaultdict(list)
        for event_type, listeners in self._topics.items():
            c._topics[event_type] = listeners[:]
        return c

    def listeners_iter(self):
        """Return an iterator over the mapping of event => listeners bound.

        NOTE(harlowja): Each listener in the yielded (event, listeners)
        tuple is an instance of the :py:class:`~.Listener`  type, which
        itself wraps a provided callback (and its details filter
        callback, if any).
        """
        for event_type, listeners in self._topics.items():
            if listeners:
                yield (event_type, listeners)

    def can_be_registered(self, event_type):
        """Checks if the event can be registered/subscribed to."""
        return True

    def can_trigger_notification(self, event_type):
        """Checks if the event can trigger a notification.

        :param event_type: event that needs to be verified
        :returns: whether the event can trigger a notification
        :rtype: boolean
        """
        if event_type in self._DISALLOWED_NOTIFICATION_EVENTS:
            return False
        else:
            return True