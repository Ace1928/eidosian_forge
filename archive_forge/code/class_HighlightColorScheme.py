import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class HighlightColorScheme(_DummyColorScheme):

    def __init__(self, theme=default_theme):
        self._code = theme['code']
        self._errmsg = theme['errmsg']
        self._filename = theme['filename']
        self._indicate = theme['indicate']
        self._highlight = theme['highlight']
        self._reset = theme['reset']
        _DummyColorScheme.__init__(self, theme=theme)

    def _markup(self, msg, color=None, style=Style.BRIGHT):
        features = ''
        if color:
            features += color
        if style:
            features += style
        with ColorShell():
            with reset_terminal() as mu:
                mu += features.encode('utf-8')
                mu += msg.encode('utf-8')
            return mu.decode('utf-8')

    def code(self, msg):
        return self._markup(msg, self._code)

    def errmsg(self, msg):
        return self._markup(msg, self._errmsg)

    def filename(self, msg):
        return self._markup(msg, self._filename)

    def indicate(self, msg):
        return self._markup(msg, self._indicate)

    def highlight(self, msg):
        return self._markup(msg, self._highlight)

    def reset(self, msg):
        return self._markup(msg, self._reset)