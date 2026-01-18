import asyncio
import collections
import logging
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.errors import (
from aiokafka.protocol.produce import ProduceRequest
from aiokafka.protocol.transaction import (
from aiokafka.structs import TopicPartition
from aiokafka.util import create_task
def _maybe_do_transactional_request(self):
    txn_manager = self._txn_manager
    tps = txn_manager.partitions_to_add()
    if tps:
        return create_task(self._do_add_partitions_to_txn(tps))
    group_id = txn_manager.consumer_group_to_add()
    if group_id is not None:
        return create_task(self._do_add_offsets_to_txn(group_id))
    commit_data = txn_manager.offsets_to_commit()
    if commit_data is not None:
        offsets, group_id = commit_data
        return create_task(self._do_txn_offset_commit(offsets, group_id))
    commit_result = txn_manager.needs_transaction_commit()
    if commit_result is not None:
        return create_task(self._do_txn_commit(commit_result))