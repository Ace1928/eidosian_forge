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
class InitPIDHandler(BaseHandler):

    def create_request(self):
        txn_manager = self._sender._txn_manager
        return InitProducerIdRequest[0](transactional_id=txn_manager.transactional_id, transaction_timeout_ms=txn_manager.transaction_timeout_ms)

    def handle_response(self, resp):
        txn_manager = self._sender._txn_manager
        error_type = Errors.for_code(resp.error_code)
        if error_type is Errors.NoError:
            log.debug('Successfully found PID=%s EPOCH=%s for Producer %s', resp.producer_id, resp.producer_epoch, self._sender.client._client_id)
            self._sender._txn_manager.set_pid_and_epoch(resp.producer_id, resp.producer_epoch)
            return
        elif error_type is CoordinatorNotAvailableError or error_type is NotCoordinatorError:
            self._sender._coordinator_dead(CoordinationType.TRANSACTION)
        elif error_type is CoordinatorLoadInProgressError or error_type is ConcurrentTransactions:
            pass
        elif error_type is TransactionalIdAuthorizationFailed:
            raise error_type(txn_manager.transactional_id)
        else:
            log.error('Unexpected error during InitProducerIdRequest: %s', error_type)
            raise error_type()
        return self._default_backoff