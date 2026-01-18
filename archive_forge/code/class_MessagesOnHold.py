import collections
import logging
import typing
from typing import Any, Callable, Iterable, Optional
class MessagesOnHold(object):
    """Tracks messages on hold by ordering key. Not thread-safe."""

    def __init__(self):
        self._size = 0
        self._messages_on_hold = collections.deque()
        self._pending_ordered_messages = {}

    @property
    def size(self) -> int:
        """Return the number of messages on hold across ordered and unordered messages.

        Note that this object may still store information about ordered messages
        in flight even if size is zero.

        Returns:
            The size value.
        """
        return self._size

    def get(self) -> Optional['subscriber.message.Message']:
        """Gets a message from the on-hold queue. A message with an ordering
        key wont be returned if there's another message with the same key in
        flight.

        Returns:
            A message that hasn't been sent to the user yet or ``None`` if there are no
            messages available.
        """
        while self._messages_on_hold:
            msg = self._messages_on_hold.popleft()
            if msg.ordering_key:
                pending_queue = self._pending_ordered_messages.get(msg.ordering_key)
                if pending_queue is None:
                    self._pending_ordered_messages[msg.ordering_key] = collections.deque()
                    self._size = self._size - 1
                    return msg
                else:
                    pending_queue.append(msg)
            else:
                self._size = self._size - 1
                return msg
        return None

    def put(self, message: 'subscriber.message.Message') -> None:
        """Put a message on hold.

        Args:
            message: The message to put on hold.
        """
        self._messages_on_hold.append(message)
        self._size = self._size + 1

    def activate_ordering_keys(self, ordering_keys: Iterable[str], schedule_message_callback: Callable[['subscriber.message.Message'], Any]) -> None:
        """Send the next message in the queue for each of the passed-in
        ordering keys, if they exist. Clean up state for keys that no longer
        have any queued messages.

        See comment at streaming_pull_manager.activate_ordering_keys() for more
        detail about the impact of this method on load.

        Args:
            ordering_keys:
                The ordering keys to activate. May be empty, or contain duplicates.
            schedule_message_callback:
                The callback to call to schedule a message to be sent to the user.
        """
        for key in ordering_keys:
            pending_ordered_messages = self._pending_ordered_messages.get(key)
            if pending_ordered_messages is None:
                _LOGGER.warning('No message queue exists for message ordering key: %s.', key)
                continue
            next_msg = self._get_next_for_ordering_key(key)
            if next_msg:
                schedule_message_callback(next_msg)
            else:
                self._clean_up_ordering_key(key)

    def _get_next_for_ordering_key(self, ordering_key: str) -> Optional['subscriber.message.Message']:
        """Get next message for ordering key.

        The client should call clean_up_ordering_key() if this method returns
        None.

        Args:
            ordering_key: Ordering key for which to get the next message.

        Returns:
            The next message for this ordering key or None if there aren't any.
        """
        queue_for_key = self._pending_ordered_messages.get(ordering_key)
        if queue_for_key:
            self._size = self._size - 1
            return queue_for_key.popleft()
        return None

    def _clean_up_ordering_key(self, ordering_key: str) -> None:
        """Clean up state for an ordering key with no pending messages.

        Args
            ordering_key: The ordering key to clean up.
        """
        message_queue = self._pending_ordered_messages.get(ordering_key)
        if message_queue is None:
            _LOGGER.warning('Tried to clean up ordering key that does not exist: %s', ordering_key)
            return
        if len(message_queue) > 0:
            _LOGGER.warning('Tried to clean up ordering key: %s with %d messages remaining.', ordering_key, len(message_queue))
            return
        del self._pending_ordered_messages[ordering_key]