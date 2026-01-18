from __future__ import annotations
import asyncio
import copy
import threading
from collections import defaultdict
from typing import (
from uuid import UUID
import jsonpatch  # type: ignore[import]
from typing_extensions import NotRequired, TypedDict
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.tracers.schemas import Run
class LogStreamCallbackHandler(BaseTracer):
    """Tracer that streams run logs to a stream."""

    def __init__(self, *, auto_close: bool=True, include_names: Optional[Sequence[str]]=None, include_types: Optional[Sequence[str]]=None, include_tags: Optional[Sequence[str]]=None, exclude_names: Optional[Sequence[str]]=None, exclude_types: Optional[Sequence[str]]=None, exclude_tags: Optional[Sequence[str]]=None, _schema_format: Literal['original', 'streaming_events']='streaming_events') -> None:
        """A tracer that streams run logs to a stream.

        Args:
            auto_close: Whether to close the stream when the root run finishes.
            include_names: Only include runs from Runnables with matching names.
            include_types: Only include runs from Runnables with matching types.
            include_tags: Only include runs from Runnables with matching tags.
            exclude_names: Exclude runs from Runnables with matching names.
            exclude_types: Exclude runs from Runnables with matching types.
            exclude_tags: Exclude runs from Runnables with matching tags.
            _schema_format: Primarily changes how the inputs and outputs are
                handled.
                **For internal use only. This API will change.**
                - 'original' is the format used by all current tracers.
                   This format is slightly inconsistent with respect to inputs
                   and outputs.
                - 'streaming_events' is used for supporting streaming events,
                   for internal usage. It will likely change in the future, or
                   be deprecated entirely in favor of a dedicated async tracer
                   for streaming events.
        """
        if _schema_format not in {'original', 'streaming_events'}:
            raise ValueError(f"Invalid schema format: {_schema_format}. Expected one of 'original', 'streaming_events'.")
        super().__init__(_schema_format=_schema_format)
        self.auto_close = auto_close
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags
        loop = asyncio.get_event_loop()
        memory_stream = _MemoryStream[RunLogPatch](loop)
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()
        self._key_map_by_run_id: Dict[UUID, str] = {}
        self._counter_map_by_name: Dict[str, int] = defaultdict(int)
        self.root_id: Optional[UUID] = None

    def __aiter__(self) -> AsyncIterator[RunLogPatch]:
        return self.receive_stream.__aiter__()

    def send(self, *ops: Dict[str, Any]) -> bool:
        """Send a patch to the stream, return False if the stream is closed."""
        self.send_stream.send_nowait(RunLogPatch(*ops))
        return True

    async def tap_output_aiter(self, run_id: UUID, output: AsyncIterator[T]) -> AsyncIterator[T]:
        """Tap an output async iterator to stream its values to the log."""
        async for chunk in output:
            if run_id != self.root_id:
                if (key := self._key_map_by_run_id.get(run_id)):
                    if not self.send({'op': 'add', 'path': f'/logs/{key}/streamed_output/-', 'value': chunk}):
                        break
            yield chunk

    def include_run(self, run: Run) -> bool:
        if run.id == self.root_id:
            return False
        run_tags = run.tags or []
        if self.include_names is None and self.include_types is None and (self.include_tags is None):
            include = True
        else:
            include = False
        if self.include_names is not None:
            include = include or run.name in self.include_names
        if self.include_types is not None:
            include = include or run.run_type in self.include_types
        if self.include_tags is not None:
            include = include or any((tag in self.include_tags for tag in run_tags))
        if self.exclude_names is not None:
            include = include and run.name not in self.exclude_names
        if self.exclude_types is not None:
            include = include and run.run_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all((tag not in self.exclude_tags for tag in run_tags))
        return include

    def _persist_run(self, run: Run) -> None:
        pass

    def _on_run_create(self, run: Run) -> None:
        """Start a run."""
        if self.root_id is None:
            self.root_id = run.id
            if not self.send({'op': 'replace', 'path': '', 'value': RunState(id=str(run.id), streamed_output=[], final_output=None, logs={}, name=run.name, type=run.run_type)}):
                return
        if not self.include_run(run):
            return
        with self.lock:
            self._counter_map_by_name[run.name] += 1
            count = self._counter_map_by_name[run.name]
            self._key_map_by_run_id[run.id] = run.name if count == 1 else f'{run.name}:{count}'
        entry = LogEntry(id=str(run.id), name=run.name, type=run.run_type, tags=run.tags or [], metadata=(run.extra or {}).get('metadata', {}), start_time=run.start_time.isoformat(timespec='milliseconds'), streamed_output=[], streamed_output_str=[], final_output=None, end_time=None)
        if self._schema_format == 'streaming_events':
            entry['inputs'] = _get_standardized_inputs(run, self._schema_format)
        self.send({'op': 'add', 'path': f'/logs/{self._key_map_by_run_id[run.id]}', 'value': entry})

    def _on_run_update(self, run: Run) -> None:
        """Finish a run."""
        try:
            index = self._key_map_by_run_id.get(run.id)
            if index is None:
                return
            ops = []
            if self._schema_format == 'streaming_events':
                ops.append({'op': 'replace', 'path': f'/logs/{index}/inputs', 'value': _get_standardized_inputs(run, self._schema_format)})
            ops.extend([{'op': 'add', 'path': f'/logs/{index}/final_output', 'value': _get_standardized_outputs(run, self._schema_format)}, {'op': 'add', 'path': f'/logs/{index}/end_time', 'value': run.end_time.isoformat(timespec='milliseconds') if run.end_time is not None else None}])
            self.send(*ops)
        finally:
            if run.id == self.root_id:
                if self.auto_close:
                    self.send_stream.close()

    def _on_llm_new_token(self, run: Run, token: str, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]]) -> None:
        """Process new LLM token."""
        index = self._key_map_by_run_id.get(run.id)
        if index is None:
            return
        self.send({'op': 'add', 'path': f'/logs/{index}/streamed_output_str/-', 'value': token}, {'op': 'add', 'path': f'/logs/{index}/streamed_output/-', 'value': chunk.message if isinstance(chunk, ChatGenerationChunk) else token})