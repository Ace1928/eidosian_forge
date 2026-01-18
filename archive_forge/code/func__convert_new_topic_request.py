import logging
import asyncio
from collections import defaultdict
from ssl import SSLContext
from typing import List, Optional, Dict, Tuple, Any
from aiokafka import __version__
from aiokafka.client import AIOKafkaClient
from aiokafka.errors import IncompatibleBrokerVersion, for_code
from aiokafka.protocol.api import Request, Response
from aiokafka.protocol.metadata import MetadataRequest
from aiokafka.protocol.commit import OffsetFetchRequest, GroupCoordinatorRequest
from aiokafka.protocol.admin import (
from aiokafka.structs import TopicPartition, OffsetAndMetadata
from .config_resource import ConfigResourceType, ConfigResource
from .new_topic import NewTopic
@staticmethod
def _convert_new_topic_request(new_topic):
    return (new_topic.name, new_topic.num_partitions, new_topic.replication_factor, [(partition_id, replicas) for partition_id, replicas in new_topic.replica_assignments.items()], [(config_key, config_value) for config_key, config_value in new_topic.topic_configs.items()])