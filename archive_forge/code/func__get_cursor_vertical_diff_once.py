from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
def _get_cursor_vertical_diff_once(self) -> int:
    """Returns the how far down the cursor moved."""
    old_top_usable_row = self.top_usable_row
    row, col = self.get_cursor_position()
    if self._last_cursor_row is None:
        cursor_dy = 0
    else:
        cursor_dy = row - self._last_cursor_row
        logger.info('cursor moved %d lines down' % cursor_dy)
        while self.top_usable_row > -1 and cursor_dy > 0:
            self.top_usable_row += 1
            cursor_dy -= 1
        while self.top_usable_row > 1 and cursor_dy < 0:
            self.top_usable_row -= 1
            cursor_dy += 1
    logger.info('top usable row changed from %d to %d', old_top_usable_row, self.top_usable_row)
    logger.info('returning cursor dy of %d from curtsies' % cursor_dy)
    self._last_cursor_row = row
    return cursor_dy