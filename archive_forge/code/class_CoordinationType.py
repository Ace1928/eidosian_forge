import asyncio
import logging
import random
import time
import aiokafka.errors as Errors
from aiokafka import __version__
from aiokafka.conn import collect_hosts, create_conn, CloseReason
from aiokafka.cluster import ClusterMetadata
from aiokafka.protocol.admin import DescribeAclsRequest_v2
from aiokafka.protocol.commit import OffsetFetchRequest
from aiokafka.protocol.coordination import FindCoordinatorRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.protocol.metadata import MetadataRequest
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.produce import ProduceRequest
from aiokafka.errors import (
from aiokafka.util import (
class CoordinationType:
    GROUP = 0
    TRANSACTION = 1