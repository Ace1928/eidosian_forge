from __future__ import annotations
import re
from typing import Callable, Dict, Generator
from ..key_binding.key_processor import KeyPress
from ..keys import Keys
from .ansi_escape_sequences import ANSI_SEQUENCES
class Vt100Parser:
    """
    Parser for VT100 input stream.
    Data can be fed through the `feed` method and the given callback will be
    called with KeyPress objects.

    ::

        def callback(key):
            pass
        i = Vt100Parser(callback)
        i.feed('data\x01...')

    :attr feed_key_callback: Function that will be called when a key is parsed.
    """

    def __init__(self, feed_key_callback: Callable[[KeyPress], None]) -> None:
        self.feed_key_callback = feed_key_callback
        self.reset()

    def reset(self, request: bool=False) -> None:
        self._in_bracketed_paste = False
        self._start_parser()

    def _start_parser(self) -> None:
        """
        Start the parser coroutine.
        """
        self._input_parser = self._input_parser_generator()
        self._input_parser.send(None)

    def _get_match(self, prefix: str) -> None | Keys | tuple[Keys, ...]:
        """
        Return the key (or keys) that maps to this prefix.
        """
        if _cpr_response_re.match(prefix):
            return Keys.CPRResponse
        elif _mouse_event_re.match(prefix):
            return Keys.Vt100MouseEvent
        try:
            return ANSI_SEQUENCES[prefix]
        except KeyError:
            return None

    def _input_parser_generator(self) -> Generator[None, str | _Flush, None]:
        """
        Coroutine (state machine) for the input parser.
        """
        prefix = ''
        retry = False
        flush = False
        while True:
            flush = False
            if retry:
                retry = False
            else:
                c = (yield)
                if isinstance(c, _Flush):
                    flush = True
                else:
                    prefix += c
            if prefix:
                is_prefix_of_longer_match = _IS_PREFIX_OF_LONGER_MATCH_CACHE[prefix]
                match = self._get_match(prefix)
                if (flush or not is_prefix_of_longer_match) and match:
                    self._call_handler(match, prefix)
                    prefix = ''
                elif (flush or not is_prefix_of_longer_match) and (not match):
                    found = False
                    retry = True
                    for i in range(len(prefix), 0, -1):
                        match = self._get_match(prefix[:i])
                        if match:
                            self._call_handler(match, prefix[:i])
                            prefix = prefix[i:]
                            found = True
                    if not found:
                        self._call_handler(prefix[0], prefix[0])
                        prefix = prefix[1:]

    def _call_handler(self, key: str | Keys | tuple[Keys, ...], insert_text: str) -> None:
        """
        Callback to handler.
        """
        if isinstance(key, tuple):
            for i, k in enumerate(key):
                self._call_handler(k, insert_text if i == 0 else '')
        elif key == Keys.BracketedPaste:
            self._in_bracketed_paste = True
            self._paste_buffer = ''
        else:
            self.feed_key_callback(KeyPress(key, insert_text))

    def feed(self, data: str) -> None:
        """
        Feed the input stream.

        :param data: Input string (unicode).
        """
        if self._in_bracketed_paste:
            self._paste_buffer += data
            end_mark = '\x1b[201~'
            if end_mark in self._paste_buffer:
                end_index = self._paste_buffer.index(end_mark)
                paste_content = self._paste_buffer[:end_index]
                self.feed_key_callback(KeyPress(Keys.BracketedPaste, paste_content))
                self._in_bracketed_paste = False
                remaining = self._paste_buffer[end_index + len(end_mark):]
                self._paste_buffer = ''
                self.feed(remaining)
        else:
            for i, c in enumerate(data):
                if self._in_bracketed_paste:
                    self.feed(data[i:])
                    break
                else:
                    self._input_parser.send(c)

    def flush(self) -> None:
        """
        Flush the buffer of the input stream.

        This will allow us to handle the escape key (or maybe meta) sooner.
        The input received by the escape key is actually the same as the first
        characters of e.g. Arrow-Up, so without knowing what follows the escape
        sequence, we don't know whether escape has been pressed, or whether
        it's something else. This flush function should be called after a
        timeout, and processes everything that's still in the buffer as-is, so
        without assuming any characters will follow.
        """
        self._input_parser.send(_Flush())

    def feed_and_flush(self, data: str) -> None:
        """
        Wrapper around ``feed`` and ``flush``.
        """
        self.feed(data)
        self.flush()