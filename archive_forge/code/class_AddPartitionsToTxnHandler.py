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
class AddPartitionsToTxnHandler(BaseHandler):
    group = ConnectionGroup.COORDINATION

    def __init__(self, sender, topic_partitions):
        super().__init__(sender)
        self._tps = topic_partitions

    def create_request(self):
        txn_manager = self._sender._txn_manager
        partition_data = collections.defaultdict(list)
        for tp in self._tps:
            partition_data[tp.topic].append(tp.partition)
        req = AddPartitionsToTxnRequest[0](transactional_id=txn_manager.transactional_id, producer_id=txn_manager.producer_id, producer_epoch=txn_manager.producer_epoch, topics=list(partition_data.items()))
        return req

    def handle_response(self, resp):
        txn_manager = self._sender._txn_manager
        unauthorized_topics = set()
        for topic, partitions in resp.errors:
            for partition, error_code in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                if error_type is Errors.NoError:
                    log.debug('Added partition %s to transaction', tp)
                    txn_manager.partition_added(tp)
                elif error_type is CoordinatorNotAvailableError or error_type is NotCoordinatorError:
                    self._sender._coordinator_dead(CoordinationType.TRANSACTION)
                    return self._default_backoff
                elif error_type is ConcurrentTransactions:
                    if not txn_manager.txn_partitions:
                        return BACKOFF_OVERRIDE
                    else:
                        return self._default_backoff
                elif error_type is CoordinatorLoadInProgressError or error_type is UnknownTopicOrPartitionError:
                    return self._default_backoff
                elif error_type is InvalidProducerEpoch:
                    raise ProducerFenced()
                elif error_type is InvalidProducerIdMapping or error_type is InvalidTxnState:
                    raise error_type()
                elif error_type is TopicAuthorizationFailedError:
                    unauthorized_topics.add(topic)
                elif error_type is OperationNotAttempted:
                    pass
                elif error_type is TransactionalIdAuthorizationFailed:
                    raise error_type(txn_manager.transactional_id)
                else:
                    log.error('Could not add partition %s due to unexpected error: %s', partition, error_type)
                    raise error_type()
        if unauthorized_topics:
            txn_manager.error_transaction(TopicAuthorizationFailedError(unauthorized_topics))
        return