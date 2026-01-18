import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
class SequenceTextWrapper(textwrap.TextWrapper):
    """Docstring overridden."""

    def __init__(self, width, term, **kwargs):
        """
        Class initializer.

        This class supports the :meth:`~.Terminal.wrap` method.
        """
        self.term = term
        textwrap.TextWrapper.__init__(self, width, **kwargs)

    def _wrap_chunks(self, chunks):
        """
        Sequence-aware variant of :meth:`textwrap.TextWrapper._wrap_chunks`.

        :raises ValueError: ``self.width`` is not a positive integer
        :rtype: list
        :returns: text chunks adjusted for width

        This simply ensures that word boundaries are not broken mid-sequence, as standard python
        textwrap would incorrectly determine the length of a string containing sequences, and may
        also break consider sequences part of a "word" that may be broken by hyphen (``-``), where
        this implementation corrects both.
        """
        lines = []
        if self.width <= 0 or not isinstance(self.width, int):
            raise ValueError('invalid width {0!r}({1!r}) (must be integer > 0)'.format(self.width, type(self.width)))
        term = self.term
        drop_whitespace = not hasattr(self, 'drop_whitespace') or self.drop_whitespace
        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            indent = self.subsequent_indent if lines else self.initial_indent
            width = self.width - len(indent)
            if drop_whitespace and (Sequence(chunks[-1], term).strip() == '' and lines):
                del chunks[-1]
            while chunks:
                chunk_len = Sequence(chunks[-1], term).length()
                if cur_len + chunk_len > width:
                    break
                cur_line.append(chunks.pop())
                cur_len += chunk_len
            if chunks and Sequence(chunks[-1], term).length() > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
            if drop_whitespace and (cur_line and Sequence(cur_line[-1], term).strip() == ''):
                del cur_line[-1]
            if cur_line:
                lines.append(indent + u''.join(cur_line))
        return lines

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        """
        Sequence-aware :meth:`textwrap.TextWrapper._handle_long_word`.

        This simply ensures that word boundaries are not broken mid-sequence, as standard python
        textwrap would incorrectly determine the length of a string containing sequences, and may
        also break consider sequences part of a "word" that may be broken by hyphen (``-``), where
        this implementation corrects both.
        """
        space_left = 1 if width < 1 else width - cur_len
        if self.break_long_words:
            term = self.term
            chunk = reversed_chunks[-1]
            idx = nxt = 0
            for text, _ in iter_parse(term, chunk):
                nxt += len(text)
                if Sequence(chunk[:nxt], term).length() > space_left:
                    break
                idx = nxt
            cur_line.append(chunk[:idx])
            reversed_chunks[-1] = chunk[idx:]
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())