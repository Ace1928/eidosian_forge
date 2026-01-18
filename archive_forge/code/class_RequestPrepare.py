import asyncio
import functools
import queue
import threading
import time
from typing import (
class RequestPrepare(NamedTuple):
    file_spec: 'CreateArtifactFileSpecInput'
    response_channel: Union['queue.Queue[ResponsePrepare]', Tuple['asyncio.AbstractEventLoop', 'asyncio.Future[ResponsePrepare]']]