import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def add_coordinator(self, node_id, host, port, rack=None, *, purpose):
    """ Keep track of all coordinator nodes separately and remove them if
        a new one was elected for the same purpose (For example group
        coordinator for group X).
        """
    if purpose in self._coordinator_by_key:
        old_id = self._coordinator_by_key.pop(purpose)
        del self._coordinators[old_id]
    self._coordinators[node_id] = BrokerMetadata(node_id, host, port, rack)
    self._coordinator_by_key[purpose] = node_id