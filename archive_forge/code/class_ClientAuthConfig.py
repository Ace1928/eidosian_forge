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
@dataclass
class ClientAuthConfig:
    client_secret_key: bytes
    client_public_key: bytes
    server_public_key: bytes