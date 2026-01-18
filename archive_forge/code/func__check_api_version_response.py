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
def _check_api_version_response(self, response):
    test_cases = [((2, 5, 0), DescribeAclsRequest_v2), ((2, 4, 0), ProduceRequest[8]), ((2, 3, 0), FetchRequest[11]), ((2, 2, 0), OffsetRequest[5]), ((2, 1, 0), FetchRequest[10]), ((2, 0, 0), FetchRequest[8]), ((1, 1, 0), FetchRequest[7]), ((1, 0, 0), MetadataRequest[5]), ((0, 11, 0), MetadataRequest[4]), ((0, 10, 2), OffsetFetchRequest[2]), ((0, 10, 1), MetadataRequest[2])]
    error_type = Errors.for_code(response.error_code)
    assert error_type is Errors.NoError, 'API version check failed'
    max_versions = {api_key: max_version for api_key, _, max_version in response.api_versions}
    for broker_version, struct in test_cases:
        if max_versions.get(struct.API_KEY, -1) >= struct.API_VERSION:
            return broker_version
    return (0, 10, 0)