import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
class InputStreamWithMemory:
    """Wraps an HTMLInputStream to remember characters since last <

    This wraps existing HTMLInputStream classes to keep track of the stream
    since the last < which marked an open tag state.

    """

    def __init__(self, inner_stream):
        self._inner_stream = inner_stream
        self.reset = self._inner_stream.reset
        self.position = self._inner_stream.position
        self._buffer = []

    @property
    def errors(self):
        return self._inner_stream.errors

    @property
    def charEncoding(self):
        return self._inner_stream.charEncoding

    @property
    def changeEncoding(self):
        return self._inner_stream.changeEncoding

    def char(self):
        c = self._inner_stream.char()
        if c:
            self._buffer.append(c)
        return c

    def charsUntil(self, characters, opposite=False):
        chars = self._inner_stream.charsUntil(characters, opposite=opposite)
        self._buffer.extend(list(chars))
        return chars

    def unget(self, char):
        if self._buffer:
            self._buffer.pop(-1)
        return self._inner_stream.unget(char)

    def get_tag(self):
        """Returns the stream history since last '<'

        Since the buffer starts at the last '<' as as seen by tagOpenState(),
        we know that everything from that point to when this method is called
        is the "tag" that is being tokenized.

        """
        return ''.join(self._buffer)

    def start_tag(self):
        """Resets stream history to just '<'

        This gets called by tagOpenState() which marks a '<' that denotes an
        open tag. Any time we see that, we reset the buffer.

        """
        self._buffer = ['<']