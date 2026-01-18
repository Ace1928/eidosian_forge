import sys
from enchant.checker import SpellChecker
@staticmethod
def _build_context(text, error_word, error_start):
    """creates the context line.

        This function will search forward and backward
        from the error word to find the nearest newlines.
        it will return this line with the error word
        colored red."""
    start_newline = text.rfind('\n', 0, error_start)
    end_newline = text.find('\n', error_start)
    return text[start_newline + 1:end_newline].replace(error_word, color(error_word, color='red'))