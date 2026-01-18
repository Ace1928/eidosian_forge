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
def _matching_api_version(self, operation: List[Request]) -> int:
    """Find the latest version of the protocol operation
        supported by both this library and the broker.

        This resolves to the lesser of either the latest api
        version this library supports, or the max version
        supported by the broker.

        :param operation: A list of protocol operation versions from
        aiokafka.protocol.
        :return: The max matching version number between client and broker.
        """
    api_key = operation[0].API_KEY
    if not self._version_info or api_key not in self._version_info:
        raise IncompatibleBrokerVersion("Kafka broker does not support the '{}' Kafka protocol.".format(operation[0].__name__))
    min_version, max_version = self._version_info[api_key]
    version = min(len(operation) - 1, max_version)
    if version < min_version:
        raise IncompatibleBrokerVersion("No version of the '{}' Kafka protocol is supported by both the client and broker.".format(operation[0].__name__))
    return version