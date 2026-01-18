import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
from ray._private.signature import extract_signature, flatten_args, recover_args
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.utils import extract_self_if_method_call
from ray.serve.exceptions import RayServeException
from ray.util.annotations import PublicAPI
class _BatchQueue:

    def __init__(self, max_batch_size: int, batch_wait_timeout_s: float, handle_batch_func: Optional[Callable]=None) -> None:
        """Async queue that accepts individual items and returns batches.

        Respects max_batch_size and timeout_s; a batch will be returned when
        max_batch_size elements are available or the timeout has passed since
        the previous get.

        If handle_batch_func is passed in, a background coroutine will run to
        poll from the queue and call handle_batch_func on the results.

        Cannot be pickled.

        Arguments:
            max_batch_size: max number of elements to return in a batch.
            timeout_s: time to wait before returning an incomplete
                batch.
            handle_batch_func(Optional[Callable]): callback to run in the
                background to handle batches if provided.
        """
        self.queue: asyncio.Queue[_SingleRequest] = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.queue_put_event = asyncio.Event()
        self._handle_batch_task = None
        if handle_batch_func is not None:
            self._handle_batch_task = get_or_create_event_loop().create_task(self._process_batches(handle_batch_func))

    def put(self, request: Tuple[_SingleRequest, asyncio.Future]) -> None:
        self.queue.put_nowait(request)
        self.queue_put_event.set()

    async def wait_for_batch(self) -> List[Any]:
        """Wait for batch respecting self.max_batch_size and self.timeout_s.

        Returns a batch of up to self.max_batch_size items. Waits for up to
        to self.timeout_s after receiving the first request that will be in
        the next batch. After the timeout, returns as many items as are ready.

        Always returns a batch with at least one item - will block
        indefinitely until an item comes in.
        """
        batch = []
        batch.append(await self.queue.get())
        max_batch_size = self.max_batch_size
        batch_wait_timeout_s = self.batch_wait_timeout_s
        batch_start_time = time.time()
        while True:
            remaining_batch_time_s = max(batch_wait_timeout_s - (time.time() - batch_start_time), 0)
            try:
                await asyncio.wait_for(self.queue_put_event.wait(), remaining_batch_time_s)
            except asyncio.TimeoutError:
                pass
            while len(batch) < max_batch_size and (not self.queue.empty()):
                batch.append(self.queue.get_nowait())
            self.queue_put_event.clear()
            if time.time() - batch_start_time >= batch_wait_timeout_s or len(batch) >= max_batch_size:
                break
        return batch

    def _validate_results(self, results: Iterable[Any], input_batch_length: int) -> None:
        if len(results) != input_batch_length:
            raise RayServeException(f"Batched function doesn't preserve batch size. The input list has length {input_batch_length} but the returned list has length {len(results)}.")

    async def _consume_func_generator(self, func_generator: AsyncGenerator, initial_futures: List[asyncio.Future], input_batch_length: int) -> None:
        """Consumes batch function generator.

        This function only runs if the function decorated with @serve.batch
        is a generator.
        """
        FINISHED_TOKEN = None
        try:
            futures = initial_futures
            async for results in func_generator:
                self._validate_results(results, input_batch_length)
                next_futures = []
                for result, future in zip(results, futures):
                    if future is FINISHED_TOKEN:
                        next_futures.append(FINISHED_TOKEN)
                    elif result in [StopIteration, StopAsyncIteration]:
                        future.set_exception(StopAsyncIteration)
                        next_futures.append(FINISHED_TOKEN)
                    else:
                        next_future = get_or_create_event_loop().create_future()
                        if not future.cancelled():
                            future.set_result(_GeneratorResult(result, next_future))
                        next_futures.append(next_future)
                futures = next_futures
            for future in futures:
                if future is not FINISHED_TOKEN:
                    future.set_exception(StopAsyncIteration)
        except Exception as e:
            for future in futures:
                if future is not FINISHED_TOKEN:
                    future.set_exception(e)

    async def _process_batches(self, func: Callable) -> None:
        """Loops infinitely and processes queued request batches."""
        while True:
            batch: List[_SingleRequest] = await self.wait_for_batch()
            assert len(batch) > 0
            self_arg = batch[0].self_arg
            args, kwargs = _batch_args_kwargs([item.flattened_args for item in batch])
            futures = [item.future for item in batch]
            if self_arg is not None:
                func_future_or_generator = func(self_arg, *args, **kwargs)
            else:
                func_future_or_generator = func(*args, **kwargs)
            if isasyncgenfunction(func):
                func_generator = func_future_or_generator
                await self._consume_func_generator(func_generator, futures, len(batch))
            else:
                try:
                    func_future = func_future_or_generator
                    results = await func_future
                    self._validate_results(results, len(batch))
                    for result, future in zip(results, futures):
                        if not future.cancelled():
                            future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)

    def __del__(self):
        if self._handle_batch_task is None or not get_or_create_event_loop().is_running():
            return
        self._handle_batch_task.cancel()