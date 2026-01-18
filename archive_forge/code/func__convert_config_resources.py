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
@classmethod
def _convert_config_resources(cls, config_resources: List[ConfigResource], op_type: str='describe') -> Tuple[Dict[int, Any], List[Any]]:
    broker_resources = defaultdict(list)
    topic_resources = []
    if op_type == 'describe':
        convert_func = cls._convert_describe_config_resource_request
    else:
        convert_func = cls._convert_alter_config_resource_request
    for config_resource in config_resources:
        resource = convert_func(config_resource)
        if config_resource.resource_type == ConfigResourceType.BROKER:
            broker_resources[int(resource[1])].append(resource)
        else:
            topic_resources.append(resource)
    return (broker_resources, topic_resources)