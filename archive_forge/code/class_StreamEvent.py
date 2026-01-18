from __future__ import annotations
from typing import Any, Dict, List
from typing_extensions import NotRequired, TypedDict
class StreamEvent(TypedDict):
    """Streaming event.

    Schema of a streaming event which is produced from the astream_events method.

    Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            async def reverse(s: str) -> str:
                return s[::-1]

            chain = RunnableLambda(func=reverse)

            events = [event async for event in chain.astream_events("hello")]

            # will produce the following events (run_id has been omitted for brevity):
            [
                {
                    "data": {"input": "hello"},
                    "event": "on_chain_start",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"chunk": "olleh"},
                    "event": "on_chain_stream",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"output": "olleh"},
                    "event": "on_chain_end",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
            ]
    """
    event: str
    'Event names are of the format: on_[runnable_type]_(start|stream|end).\n    \n    Runnable types are one of: \n    * llm - used by non chat models\n    * chat_model - used by chat models\n    * prompt --  e.g., ChatPromptTemplate\n    * tool -- from tools defined via @tool decorator or inheriting from Tool/BaseTool\n    * chain - most Runnables are of this type\n    \n    Further, the events are categorized as one of:\n    * start - when the runnable starts\n    * stream - when the runnable is streaming\n    * end - when the runnable ends\n    \n    start, stream and end are associated with slightly different `data` payload.\n    \n    Please see the documentation for `EventData` for more details.\n    '
    name: str
    'The name of the runnable that generated the event.'
    run_id: str
    'An randomly generated ID to keep track of the execution of the given runnable.\n    \n    Each child runnable that gets invoked as part of the execution of a parent runnable\n    is assigned its own unique ID.\n    '
    tags: NotRequired[List[str]]
    'Tags associated with the runnable that generated this event.\n    \n    Tags are always inherited from parent runnables.\n    \n    Tags can either be bound to a runnable using `.with_config({"tags":  ["hello"]})`\n    or passed at run time using `.astream_events(..., {"tags": ["hello"]})`.\n    '
    metadata: NotRequired[Dict[str, Any]]
    'Metadata associated with the runnable that generated this event.\n    \n    Metadata can either be bound to a runnable using \n    \n        `.with_config({"metadata": { "foo": "bar" }})`\n        \n    or passed at run time using \n    \n        `.astream_events(..., {"metadata": {"foo": "bar"}})`.\n    '
    data: EventData
    'Event data.\n\n    The contents of the event data depend on the event type.\n    '