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
def _convert_describe_config_resource_request(config_resource):
    return (config_resource.resource_type, config_resource.name, list(config_resource.configs.keys()) if config_resource.configs else None)