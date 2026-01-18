from __future__ import annotations
import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Callable, Iterable, Iterator, cast
from typing_extensions import Awaitable, AsyncIterable, AsyncIterator, assert_never
import httpx
from ..._utils import is_dict, is_list, consume_sync_iterator, consume_async_iterator
from ..._models import construct_type
from ..._streaming import Stream, AsyncStream
from ...types.beta import AssistantStreamEvent
from ...types.beta.threads import (
from ...types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta
def accumulate_delta(acc: dict[object, object], delta: dict[object, object]) -> dict[object, object]:
    for key, delta_value in delta.items():
        if key not in acc:
            acc[key] = delta_value
            continue
        acc_value = acc[key]
        if acc_value is None:
            acc[key] = delta_value
            continue
        if key == 'index' or key == 'type':
            acc[key] = delta_value
            continue
        if isinstance(acc_value, str) and isinstance(delta_value, str):
            acc_value += delta_value
        elif isinstance(acc_value, (int, float)) and isinstance(delta_value, (int, float)):
            acc_value += delta_value
        elif is_dict(acc_value) and is_dict(delta_value):
            acc_value = accumulate_delta(acc_value, delta_value)
        elif is_list(acc_value) and is_list(delta_value):
            if all((isinstance(x, (str, int, float)) for x in acc_value)):
                acc_value.extend(delta_value)
                continue
            for delta_entry in delta_value:
                if not is_dict(delta_entry):
                    raise TypeError(f'Unexpected list delta entry is not a dictionary: {delta_entry}')
                try:
                    index = delta_entry['index']
                except KeyError as exc:
                    raise RuntimeError(f'Expected list delta entry to have an `index` key; {delta_entry}') from exc
                if not isinstance(index, int):
                    raise TypeError(f'Unexpected, list delta entry `index` value is not an integer; {index}')
                try:
                    acc_entry = acc_value[index]
                except IndexError:
                    acc_value.insert(index, delta_entry)
                else:
                    if not is_dict(acc_entry):
                        raise TypeError('not handled yet')
                    acc_value[index] = accumulate_delta(acc_entry, delta_entry)
        acc[key] = acc_value
    return acc