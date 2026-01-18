import json
import time
import asyncio
import typing
import threading
from contextlib import asynccontextmanager, suppress
from lazyops.utils.logs import default_logger as logger
from aiokeydb.v1.client.types import KeyDBUri, lazyproperty
from lazyops.utils.serialization import ObjectEncoder
from aiokeydb.v1.client.serializers import SerializerType
from aiokeydb.v1.client.meta import KeyDBClient
from aiokeydb.v1.client.schemas.session import KeyDBSession
from aiokeydb.v1.connection import ConnectionPool, BlockingConnectionPool
from aiokeydb.v1.commands.core import AsyncScript
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool, AsyncBlockingConnectionPool
from aiokeydb.v1.queues.errors import JobError
from aiokeydb.v1.queues.types import (
from aiokeydb.v1.queues.utils import (
from lazyops.imports._aiohttpx import (
from aiokeydb.v1.utils import set_ulimits, get_ulimits
def _set_silenced_functions(self):
    from aiokeydb.v1.queues.worker import WorkerTasks
    self.silenced_functions = list(set(WorkerTasks.silenced_functions))