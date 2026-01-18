import asyncio
import collections
import logging
import random
import time
from itertools import chain
import async_timeout
import aiokafka.errors as Errors
from aiokafka.errors import (
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.record.memory_records import MemoryRecords
from aiokafka.record.control_record import ControlRecord, ABORT_MARKER
from aiokafka.structs import OffsetAndTimestamp, TopicPartition, ConsumerRecord
from aiokafka.util import create_future, create_task
def _get_actions_per_node(self, assignment):
    """ For each assigned partition determine the action needed to be
        performed and group those by leader node id.
        """
    fetchable = collections.defaultdict(list)
    awaiting_reset = collections.defaultdict(list)
    backoff_by_nodes = collections.defaultdict(list)
    resume_futures = []
    invalid_metadata = False
    for tp in assignment.tps:
        tp_state = assignment.state_value(tp)
        node_id = self._client.cluster.leader_for_partition(tp)
        backoff = 0
        if tp in self._records:
            record = self._records[tp]
            backoff = record.calculate_backoff()
            if backoff:
                backoff_by_nodes[node_id].append(backoff)
        elif node_id in self._in_flight:
            continue
        elif node_id is None or node_id == -1:
            log.debug('No leader found for partition %s. Waiting metadata update', tp)
            invalid_metadata = True
        elif not tp_state.has_valid_position:
            awaiting_reset[node_id].append(tp)
        elif tp_state.paused:
            resume_futures.append(tp_state.resume_fut)
        else:
            position = tp_state.position
            fetchable[node_id].append((tp, position))
            log.debug('Adding fetch request for partition %s at offset %d', tp, position)
    fetch_requests = []
    for node_id, partition_data in fetchable.items():
        if node_id in backoff_by_nodes:
            continue
        if node_id in awaiting_reset:
            continue
        random.shuffle(partition_data)
        by_topics = collections.defaultdict(list)
        for tp, position in partition_data:
            by_topics[tp.topic].append((tp.partition, position, self._max_partition_fetch_bytes))
        klass = self._fetch_request_class
        if klass.API_VERSION > 3:
            req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, self._fetch_max_bytes, self._isolation_level, list(by_topics.items()))
        elif klass.API_VERSION == 3:
            req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, self._fetch_max_bytes, list(by_topics.items()))
        else:
            req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, list(by_topics.items()))
        fetch_requests.append((node_id, req))
    if backoff_by_nodes:
        backoff = min(map(max, backoff_by_nodes.values()))
    else:
        backoff = self._fetcher_timeout
    return (fetch_requests, awaiting_reset, backoff, invalid_metadata, resume_futures)