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
def _convert_topic_partitions(topic_partitions: Dict[str, TopicPartition]):
    return [(topic_name, (new_part.total_count, new_part.new_assignments)) for topic_name, new_part in topic_partitions.items()]