import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
def _live_receivers(self, sender):
    """
        Filter sequence of receivers to get resolved, live receivers.

        This checks for weak references and resolves them, then returning only
        live receivers.
        """
    receivers = None
    if self.use_caching and (not self._dead_receivers):
        receivers = self.sender_receivers_cache.get(sender)
        if receivers is NO_RECEIVERS:
            return ([], [])
    if receivers is None:
        with self.lock:
            self._clear_dead_receivers()
            senderkey = _make_id(sender)
            receivers = []
            for (_receiverkey, r_senderkey), receiver, is_async in self.receivers:
                if r_senderkey == NONE_ID or r_senderkey == senderkey:
                    receivers.append((receiver, is_async))
            if self.use_caching:
                if not receivers:
                    self.sender_receivers_cache[sender] = NO_RECEIVERS
                else:
                    self.sender_receivers_cache[sender] = receivers
    non_weak_sync_receivers = []
    non_weak_async_receivers = []
    for receiver, is_async in receivers:
        if isinstance(receiver, weakref.ReferenceType):
            receiver = receiver()
            if receiver is not None:
                if is_async:
                    non_weak_async_receivers.append(receiver)
                else:
                    non_weak_sync_receivers.append(receiver)
        elif is_async:
            non_weak_async_receivers.append(receiver)
        else:
            non_weak_sync_receivers.append(receiver)
    return (non_weak_sync_receivers, non_weak_async_receivers)