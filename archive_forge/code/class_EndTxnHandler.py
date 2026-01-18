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
class EndTxnHandler(BaseHandler):
    group = ConnectionGroup.COORDINATION

    def __init__(self, sender, commit_result):
        super().__init__(sender)
        self._commit_result = commit_result

    def create_request(self):
        txn_manager = self._sender._txn_manager
        req = EndTxnRequest[0](transactional_id=txn_manager.transactional_id, producer_id=txn_manager.producer_id, producer_epoch=txn_manager.producer_epoch, transaction_result=self._commit_result)
        return req

    def handle_response(self, resp):
        txn_manager = self._sender._txn_manager
        error_type = Errors.for_code(resp.error_code)
        if error_type is Errors.NoError:
            txn_manager.complete_transaction()
            return
        elif error_type is CoordinatorNotAvailableError or error_type is NotCoordinatorError:
            self._sender._coordinator_dead(CoordinationType.TRANSACTION)
        elif error_type is CoordinatorLoadInProgressError or error_type is ConcurrentTransactions:
            pass
        elif error_type is InvalidProducerEpoch:
            raise ProducerFenced()
        elif error_type is InvalidTxnState:
            raise error_type()
        else:
            log.error('Could not end transaction due to unexpected error: %s', error_type)
            raise error_type()
        return self._default_backoff