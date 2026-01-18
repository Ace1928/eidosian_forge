import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def _generate_bootstrap_brokers(self):
    bootstrap_hosts = collect_hosts(self.config['bootstrap_servers'])
    brokers = {}
    for i, (host, port, _) in enumerate(bootstrap_hosts):
        node_id = 'bootstrap-%s' % i
        brokers[node_id] = BrokerMetadata(node_id, host, port, None)
    return brokers