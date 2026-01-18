import collections
import logging
import threading
from typing import Callable, Optional, Type
import grpc
from grpc import _common
from grpc._cython import cygrpc
from grpc._typing import MetadataType
class _CallbackState(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.called = False
        self.exception = None