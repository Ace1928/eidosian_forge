import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class HTMLChunker(Chunker):
    """Chunker for breaking up HTML documents into chunks of checkable text.

    The operation of this chunker is very simple - anything between a "<"
    and a ">" will be ignored.  Later versions may improve the algorithm
    slightly.
    """

    def next(self):
        text = self._text
        offset = self.offset
        while True:
            if offset >= len(text):
                break
            if text[offset] == '<':
                maybe_tag = offset
                if self._is_tag(text, offset):
                    while text[offset] != '>':
                        offset += 1
                        if offset == len(text):
                            offset = maybe_tag + 1
                            break
                    else:
                        offset += 1
                else:
                    offset = maybe_tag + 1
            s_pos = offset
            while offset < len(text) and text[offset] != '<':
                offset += 1
            self._offset = offset
            if s_pos < offset:
                return (text[s_pos:offset], s_pos)
        raise StopIteration()

    def _is_tag(self, text, offset):
        if offset + 1 < len(text):
            if text[offset + 1].isalpha():
                return True
            if text[offset + 1] == '/':
                return True
        return False