from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import Filter, to_filter
class ThreadedAutoSuggest(AutoSuggest):
    """
    Wrapper that runs auto suggestions in a thread.
    (Use this to prevent the user interface from becoming unresponsive if the
    generation of suggestions takes too much time.)
    """

    def __init__(self, auto_suggest: AutoSuggest) -> None:
        self.auto_suggest = auto_suggest

    def get_suggestion(self, buff: Buffer, document: Document) -> Suggestion | None:
        return self.auto_suggest.get_suggestion(buff, document)

    async def get_suggestion_async(self, buff: Buffer, document: Document) -> Suggestion | None:
        """
        Run the `get_suggestion` function in a thread.
        """

        def run_get_suggestion_thread() -> Suggestion | None:
            return self.get_suggestion(buff, document)
        return await run_in_executor_with_context(run_get_suggestion_thread)