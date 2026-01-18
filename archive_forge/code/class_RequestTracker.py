import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
from vllm.lora.request import LoRARequest
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self, exc: Exception, request_id: Optional[str]=None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self, request_output: RequestOutput, *, verbose: bool=False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f'Finished request {request_id}.')
            self.abort_request(request_id)

    def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f'Request {request_id} already exists.')
        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {'request_id': request_id, **engine_add_request_kwargs}))
        self.new_requests_event.set()
        return stream

    def abort_request(self, request_id: str, *, verbose: bool=False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f'Aborted request {request_id}.')
        self._finished_requests.put_nowait(request_id)
        if request_id not in self._request_streams or self._request_streams[request_id].finished:
            return
        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()
        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)
        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)
        self.new_requests_event.clear()
        return (new_requests, finished_requests)

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()