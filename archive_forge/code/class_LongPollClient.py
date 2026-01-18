import asyncio
import logging
import os
import random
from asyncio.events import AbstractEventLoop
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, DefaultDict, Dict, Optional, Set, Tuple, Union
import ray
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.common import ReplicaName
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import format_actor_name
from ray.serve.generated.serve_pb2 import ActorNameList
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet, LongPollRequest, LongPollResult
from ray.serve.generated.serve_pb2 import UpdatedObject as UpdatedObjectProto
from ray.util import metrics
class LongPollClient:
    """The asynchronous long polling client.

    Args:
        host_actor: handle to actor embedding LongPollHost.
        key_listeners: a dictionary mapping keys to
          callbacks to be called on state update for the corresponding keys.
        call_in_event_loop: an asyncio event loop
          to post the callback into.
    """

    def __init__(self, host_actor, key_listeners: Dict[KeyType, UpdateStateCallable], call_in_event_loop: AbstractEventLoop) -> None:
        assert len(key_listeners) > 0
        assert call_in_event_loop is not None
        self.host_actor = host_actor
        self.key_listeners = key_listeners
        self.event_loop = call_in_event_loop
        self.snapshot_ids: Dict[KeyType, int] = {key: -1 for key in self.key_listeners.keys()}
        self.is_running = True
        self._poll_next()

    def _on_callback_completed(self, trigger_at: int):
        """Called after a single callback is completed.

        When the total number of callback completed equals to trigger_at,
        _poll_next() will be called. This is designed to make sure we only
        _poll_next() after all the state callbacks completed. This is a
        way to serialize the callback invocations between object versions.
        """
        self._callbacks_processed_count += 1
        if self._callbacks_processed_count == trigger_at:
            self._poll_next()

    def _poll_next(self):
        """Poll the update. The callback is expected to scheduler another
        _poll_next call.
        """
        self._callbacks_processed_count = 0
        self._current_ref = self.host_actor.listen_for_change.remote(self.snapshot_ids)
        self._current_ref._on_completed(lambda update: self._process_update(update))

    def _schedule_to_event_loop(self, callback):
        if self.event_loop.is_running():
            self.event_loop.call_soon_threadsafe(callback)
        else:
            logger.error('The event loop is closed, shutting down long poll client.')
            self.is_running = False

    def _process_update(self, updates: Dict[str, UpdatedObject]):
        if isinstance(updates, ray.exceptions.RayActorError):
            logger.debug('LongPollClient failed to connect to host. Shutting down.')
            self.is_running = False
            return
        if isinstance(updates, ConnectionError):
            logger.warning('LongPollClient connection failed, shutting down.')
            self.is_running = False
            return
        if isinstance(updates, ray.exceptions.RayTaskError):
            logger.error('LongPollHost errored\n' + updates.traceback_str)
            self._schedule_to_event_loop(self._poll_next)
            return
        if updates == LongPollState.TIME_OUT:
            logger.debug('LongPollClient polling timed out. Retrying.')
            self._schedule_to_event_loop(self._poll_next)
            return
        logger.debug(f'LongPollClient {self} received updates for keys: {list(updates.keys())}.', extra={'log_to_stderr': False})
        for key, update in updates.items():
            self.snapshot_ids[key] = update.snapshot_id
            callback = self.key_listeners[key]

            def chained(callback=callback, arg=update.object_snapshot):
                callback(arg)
                self._on_callback_completed(trigger_at=len(updates))
            self._schedule_to_event_loop(chained)