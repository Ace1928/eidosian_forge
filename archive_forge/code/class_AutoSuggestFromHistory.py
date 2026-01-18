from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from .filters import to_cli_filter
class AutoSuggestFromHistory(AutoSuggest):
    """
    Give suggestions based on the lines in the history.
    """

    def get_suggestion(self, cli, buffer, document):
        history = buffer.history
        text = document.text.rsplit('\n', 1)[-1]
        if text.strip():
            for string in reversed(list(history)):
                for line in reversed(string.splitlines()):
                    if line.startswith(text):
                        return Suggestion(line[len(text):])