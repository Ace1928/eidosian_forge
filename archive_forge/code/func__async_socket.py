import asyncio
import logging
import sys
import time
from typing import Dict, Optional, Union
from warnings import warn
import zmq
import zmq.asyncio
from rpcq._base import to_msgpack, from_msgpack
import rpcq._utils as utils
from rpcq.messages import RPCError, RPCReply
@property
def _async_socket(self):
    """
        Creates a new async socket if one doesn't already exist for this Client
        """
    if not self._async_socket_cache:
        self._async_socket_cache = self._connect_to_socket(zmq.asyncio.Context(), self.endpoint)
    return self._async_socket_cache