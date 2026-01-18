from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
class ThreadedCompleter(Completer):
    """
    Wrapper that runs the `get_completions` generator in a thread.

    (Use this to prevent the user interface from becoming unresponsive if the
    generation of completions takes too much time.)

    The completions will be displayed as soon as they are produced. The user
    can already select a completion, even if not all completions are displayed.
    """

    def __init__(self, completer: Completer) -> None:
        self.completer = completer

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        return self.completer.get_completions(document, complete_event)

    async def get_completions_async(self, document: Document, complete_event: CompleteEvent) -> AsyncGenerator[Completion, None]:
        """
        Asynchronous generator of completions.
        """
        async with aclosing(generator_to_async_generator(lambda: self.completer.get_completions(document, complete_event))) as async_generator:
            async for completion in async_generator:
                yield completion

    def __repr__(self) -> str:
        return f'ThreadedCompleter({self.completer!r})'