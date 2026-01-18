import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class KivyFormatter(logging.Formatter):
    """Split out first field in message marked with a colon,
    and either apply terminal color codes to the record, or strip
    out color markup if colored logging is not available.

    .. versionadded:: 2.2.0"""

    def __init__(self, *args, use_color=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._coloring_cls = ColoredLogRecord if use_color else UncoloredLogRecord

    def format(self, record):
        return super().format(self._coloring_cls(ColonSplittingLogRecord(record)))