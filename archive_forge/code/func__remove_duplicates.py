from __future__ import unicode_literals
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from .compiler import _CompiledGrammar
def _remove_duplicates(self, items):
    """
        Remove duplicates, while keeping the order.
        (Sometimes we have duplicates, because the there several matches of the
        same grammar, each yielding similar completions.)
        """
    result = []
    for i in items:
        if i not in result:
            result.append(i)
    return result