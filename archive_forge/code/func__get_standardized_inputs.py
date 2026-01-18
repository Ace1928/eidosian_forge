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
def _get_standardized_inputs(run: Run, schema_format: Literal['original', 'streaming_events']) -> Optional[Dict[str, Any]]:
    """Extract standardized inputs from a run.

    Standardizes the inputs based on the type of the runnable used.

    Args:
        run: Run object
        schema_format: The schema format to use.

    Returns:
        Valid inputs are only dict. By conventions, inputs always represented
        invocation using named arguments.
        A None means that the input is not yet known!
    """
    if schema_format == 'original':
        raise NotImplementedError('Do not assign inputs with original schema drop the key for now.When inputs are added to astream_log they should be added with standardized schema for streaming events.')
    inputs = load(run.inputs)
    if run.run_type in {'retriever', 'llm', 'chat_model'}:
        return inputs
    inputs = inputs['input']
    if inputs == {'input': ''}:
        return None
    return inputs