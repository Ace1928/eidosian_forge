from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def _add_keys(self) -> None:
    """Populate the keyboard with key instances

        Set the _keys and rect attributes.

        """
    key_map: list[Key | Literal[None]] = [None] * 128
    start_note = self._start_note
    end_note = self._end_note
    black_offset = self.black_key_width // 2
    prev_white_key = None
    x = y = 0
    if is_white_key(start_note):
        is_prev_white = True
    else:
        x += black_offset
        is_prev_white = False
    for note in range(start_note, end_note + 1):
        ident = note
        if is_white_key(note):
            if is_prev_white:
                if note == end_note or is_white_key(note + 1):
                    key = self.WhiteKey(ident, (x, y), prev_white_key)
                else:
                    key = self.WhiteKeyLeft(ident, (x, y), prev_white_key)
            elif note == end_note or is_white_key(note + 1):
                key = self.WhiteKeyRight(ident, (x, y), prev_white_key)
            else:
                key = self.WhiteKeyCenter(ident, (x, y), prev_white_key)
            is_prev_white = True
            x += self.white_key_width
            prev_white_key = key
        else:
            key = self.BlackKey(ident, (x - black_offset, y), prev_white_key)
            is_prev_white = False
        key_map[note] = key
    self._keys = key_map
    the_key = key_map[self._end_note]
    if the_key is None:
        kb_width = 0
    else:
        kb_width = the_key.rect.right
    kb_height = self.white_key_height
    self.rect = pg.Rect(0, 0, kb_width, kb_height)