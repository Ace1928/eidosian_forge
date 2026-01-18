import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class ConsoleHandler(logging.StreamHandler):
    """
        Emits records to a stream (by default, stderr).

        However, if the msg starts with "stderr:" it is not formatted, but
        written straight to the stream.

        .. versionadded:: 2.2.0
    """

    def filter(self, record):
        try:
            msg = record.msg
            k = msg.split(':', 1)
            if k[0] == 'stderr' and len(k) == 2:
                self.stream.write(k[1] + '\n')
                return False
        except Exception:
            pass
        return True