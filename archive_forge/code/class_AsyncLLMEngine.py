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
class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        *kwargs: Arguments for LLMEngine.
    """
    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self, worker_use_ray: bool, engine_use_ray: bool, *args, log_requests: bool=True, max_log_len: Optional[int]=None, start_engine_loop: bool=True, **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)
        self.background_loop = None
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return self.background_loop is not None and (not self.background_loop.done())

    def get_tokenizer(self):
        return self.engine.tokenizer.tokenizer

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError('Background loop is already running.')
        self._request_tracker.init_event()
        self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(partial(_raise_exception_on_finish, request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args, **kwargs) -> Union[_AsyncLLMEngine, 'ray.ObjectRef']:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            cache_config = args[1]
            parallel_config = args[2]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""
        new_requests, finished_requests = self._request_tracker.get_new_and_finished_requests()
        for new_request in new_requests:
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                await self.engine.add_request_async(**new_request)
        if finished_requests:
            await self._engine_abort(finished_requests)
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()
        for request_output in request_outputs:
            self._request_tracker.process_request_output(request_output, verbose=self.log_requests)
        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(self, request_id: str, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]]=None, arrival_time: Optional[float]=None, lora_request: Optional[LoRARequest]=None, prefix_pos: Optional[int]=None) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.max_log_len]
            logger.info(f'Received request {request_id}: prompt: {shortened_prompt!r}, prefix_pos: {prefix_pos},sampling_params: {sampling_params}, prompt_token_ids: {shortened_token_ids}, lora_request: {lora_request}.')
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError('Background loop is not running. If it was running, inspect the output to find the stacktrace of the error that caused the background loop to stop (AsyncEngineDeadError).')
        if arrival_time is None:
            arrival_time = time.time()
        if self.engine_use_ray:
            prompt_token_ids = await self.engine.encode_request_async.remote(request_id=request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, lora_request=lora_request)
        else:
            prompt_token_ids = await self.engine.encode_request_async(request_id=request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, lora_request=lora_request)
        stream = self._request_tracker.add_request(request_id, prompt=prompt, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, arrival_time=arrival_time, lora_request=lora_request, prefix_pos=prefix_pos)
        return stream

    async def generate(self, prompt: Optional[str], sampling_params: SamplingParams, request_id: str, prompt_token_ids: Optional[List[int]]=None, lora_request: Optional[LoRARequest]=None, prefix_pos: Optional[int]=None) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            lora_request: LoRA request to use for generation, if any.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        arrival_time = time.monotonic()
        try:
            stream = await self.add_request(request_id, prompt, sampling_params, prompt_token_ids=prompt_token_ids, arrival_time=arrival_time, lora_request=lora_request, prefix_pos=prefix_pos)
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError('Background loop is not running. If it was running, inspect the output to find the stacktrace of the error that caused the background loop to stop (AsyncEngineDeadError).')
        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id, verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs, start_engine_loop: bool=True) -> 'AsyncLLMEngine':
        """Creates an async LLM engine from the engine arguments."""
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        placement_group = initialize_cluster(parallel_config, engine_args.engine_use_ray)
        engine = cls(parallel_config.worker_use_ray, engine_args.engine_use_ray, *engine_configs, placement_group, log_requests=not engine_args.disable_log_requests, log_stats=not engine_args.disable_log_stats, max_log_len=engine_args.max_log_len, start_engine_loop=start_engine_loop)
        return engine

    async def do_log_stats(self) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote()
        else:
            self.engine.do_log_stats()