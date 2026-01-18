from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
class ConditionalCompleter(Completer):
    """
    Wrapper around any other completer that will enable/disable the completions
    depending on whether the received condition is satisfied.

    :param completer: :class:`.Completer` instance.
    :param filter: :class:`.Filter` instance.
    """

    def __init__(self, completer: Completer, filter: FilterOrBool) -> None:
        self.completer = completer
        self.filter = to_filter(filter)

    def __repr__(self) -> str:
        return f'ConditionalCompleter({self.completer!r}, filter={self.filter!r})'

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        if self.filter():
            yield from self.completer.get_completions(document, complete_event)

    async def get_completions_async(self, document: Document, complete_event: CompleteEvent) -> AsyncGenerator[Completion, None]:
        if self.filter():
            async with aclosing(self.completer.get_completions_async(document, complete_event)) as async_generator:
                async for item in async_generator:
                    yield item