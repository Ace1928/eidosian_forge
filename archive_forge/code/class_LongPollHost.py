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
class LongPollHost:
    """The server side object that manages long pulling requests.

    The desired use case is to embed this in an Ray actor. Client will be
    expected to call actor.listen_for_change.remote(...). On the host side,
    you can call host.notify_changed(key, object) to update the state and
    potentially notify whoever is polling for these values.

    Internally, we use snapshot_ids for each object to identify client with
    outdated object and immediately return the result. If the client has the
    up-to-date verison, then the listen_for_change call will only return when
    the object is updated.
    """

    def __init__(self, listen_for_change_request_timeout_s: Tuple[int, int]=LISTEN_FOR_CHANGE_REQUEST_TIMEOUT_S):
        self.snapshot_ids: DefaultDict[KeyType, int] = defaultdict(lambda: random.randint(0, 1000000))
        self.object_snapshots: Dict[KeyType, Any] = dict()
        self.notifier_events: DefaultDict[KeyType, Set[asyncio.Event]] = defaultdict(set)
        self._listen_for_change_request_timeout_s = listen_for_change_request_timeout_s
        self.transmission_counter = metrics.Counter('serve_long_poll_host_transmission_counter', description='The number of times the long poll host transmits data.', tag_keys=('namespace_or_state',))

    def _get_num_notifier_events(self, key: Optional[KeyType]=None):
        """Used for testing."""
        if key is not None:
            return len(self.notifier_events[key])
        else:
            return sum((len(events) for events in self.notifier_events.values()))

    def _count_send(self, timeout_or_data: Union[LongPollState, Dict[KeyType, UpdatedObject]]):
        """Helper method that tracks the data sent by listen_for_change.

        Records number of times long poll host sends data in the
        ray_serve_long_poll_host_send_counter metric.
        """
        if isinstance(timeout_or_data, LongPollState):
            self.transmission_counter.inc(value=1, tags={'namespace_or_state': 'TIMEOUT'})
        else:
            data = timeout_or_data
            for key in data.keys():
                self.transmission_counter.inc(value=1, tags={'namespace_or_state': str(key)})

    async def listen_for_change(self, keys_to_snapshot_ids: Dict[KeyType, int]) -> Union[LongPollState, Dict[KeyType, UpdatedObject]]:
        """Listen for changed objects.

        This method will returns a dictionary of updated objects. It returns
        immediately if the snapshot_ids are outdated, otherwise it will block
        until there's an update.
        """
        watched_keys = keys_to_snapshot_ids.keys()
        existent_keys = set(watched_keys).intersection(set(self.snapshot_ids.keys()))
        updated_objects = {key: UpdatedObject(self.object_snapshots[key], self.snapshot_ids[key]) for key in existent_keys if self.snapshot_ids[key] != keys_to_snapshot_ids[key]}
        if len(updated_objects) > 0:
            self._count_send(updated_objects)
            return updated_objects
        async_task_to_events = {}
        async_task_to_watched_keys = {}
        for key in watched_keys:
            event = asyncio.Event()
            self.notifier_events[key].add(event)
            task = get_or_create_event_loop().create_task(event.wait())
            async_task_to_events[task] = event
            async_task_to_watched_keys[task] = key
        done, not_done = await asyncio.wait(async_task_to_watched_keys.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=random.uniform(*self._listen_for_change_request_timeout_s))
        for task in not_done:
            task.cancel()
            try:
                event = async_task_to_events[task]
                self.notifier_events[async_task_to_watched_keys[task]].remove(event)
            except KeyError:
                pass
        if len(done) == 0:
            self._count_send(LongPollState.TIME_OUT)
            return LongPollState.TIME_OUT
        else:
            updated_object_key: str = async_task_to_watched_keys[done.pop()]
            updated_object = {updated_object_key: UpdatedObject(self.object_snapshots[updated_object_key], self.snapshot_ids[updated_object_key])}
            self._count_send(updated_object)
            return updated_object

    async def listen_for_change_java(self, keys_to_snapshot_ids_bytes: bytes) -> bytes:
        """Listen for changed objects. only call by java proxy/router now.
        Args:
            keys_to_snapshot_ids_bytes (Dict[str, int]): the protobuf bytes of
              keys_to_snapshot_ids (Dict[str, int]).
        """
        request_proto = LongPollRequest.FromString(keys_to_snapshot_ids_bytes)
        keys_to_snapshot_ids = {self._parse_xlang_key(xlang_key): snapshot_id for xlang_key, snapshot_id in request_proto.keys_to_snapshot_ids.items()}
        keys_to_updated_objects = await self.listen_for_change(keys_to_snapshot_ids)
        return self._listen_result_to_proto_bytes(keys_to_updated_objects)

    def _parse_poll_namespace(self, name: str):
        if name == LongPollNamespace.ROUTE_TABLE.name:
            return LongPollNamespace.ROUTE_TABLE
        elif name == LongPollNamespace.RUNNING_REPLICAS.name:
            return LongPollNamespace.RUNNING_REPLICAS
        else:
            return name

    def _parse_xlang_key(self, xlang_key: str) -> KeyType:
        if xlang_key is None:
            raise ValueError('func _parse_xlang_key: xlang_key is None')
        if xlang_key.startswith('(') and xlang_key.endswith(')'):
            fields = xlang_key[1:-1].split(',')
            if len(fields) == 2:
                enum_field = self._parse_poll_namespace(fields[0].strip())
                if isinstance(enum_field, LongPollNamespace):
                    return (enum_field, fields[1].strip())
        else:
            return self._parse_poll_namespace(xlang_key)
        raise ValueError('can not parse key type from xlang_key {}'.format(xlang_key))

    def _build_xlang_key(self, key: KeyType) -> str:
        if isinstance(key, tuple):
            return '(' + key[0].name + ',' + key[1] + ')'
        elif isinstance(key, LongPollNamespace):
            return key.name
        else:
            return key

    def _object_snapshot_to_proto_bytes(self, key: KeyType, object_snapshot: Any) -> bytes:
        if key == LongPollNamespace.ROUTE_TABLE:
            xlang_endpoints = {str(endpoint_tag): EndpointInfoProto(route=endpoint_info.route) for endpoint_tag, endpoint_info in object_snapshot.items()}
            return EndpointSet(endpoints=xlang_endpoints).SerializeToString()
        elif isinstance(key, tuple) and key[0] == LongPollNamespace.RUNNING_REPLICAS:
            actor_name_list = [f'{ReplicaName.prefix}{format_actor_name(replica_info.replica_tag)}' for replica_info in object_snapshot]
            return ActorNameList(names=actor_name_list).SerializeToString()
        else:
            return str.encode(str(object_snapshot))

    def _listen_result_to_proto_bytes(self, keys_to_updated_objects: Dict[KeyType, UpdatedObject]) -> bytes:
        xlang_keys_to_updated_objects = {self._build_xlang_key(key): UpdatedObjectProto(snapshot_id=updated_object.snapshot_id, object_snapshot=self._object_snapshot_to_proto_bytes(key, updated_object.object_snapshot)) for key, updated_object in keys_to_updated_objects.items()}
        data = {'updated_objects': xlang_keys_to_updated_objects}
        proto = LongPollResult(**data)
        return proto.SerializeToString()

    def notify_changed(self, object_key: KeyType, updated_object: Any):
        self.snapshot_ids[object_key] += 1
        self.object_snapshots[object_key] = updated_object
        logger.debug(f'LongPollHost: Notify change for key {object_key}.')
        if object_key in self.notifier_events:
            for event in self.notifier_events.pop(object_key):
                event.set()