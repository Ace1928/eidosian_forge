import collections
import logging
import threading
from typing import Callable, Optional, Type
import grpc
from grpc import _common
from grpc._cython import cygrpc
from grpc._typing import MetadataType
class _AuthMetadataContext(collections.namedtuple('AuthMetadataContext', ('service_url', 'method_name')), grpc.AuthMetadataContext):
    pass