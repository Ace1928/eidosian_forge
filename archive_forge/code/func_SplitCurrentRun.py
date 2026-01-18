import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def SplitCurrentRun(self, textPosition):
    if not self._current_run:
        return
    if textPosition <= self._current_run.text_start:
        return
    new_run = copy.copy(self._current_run)
    new_run.next_run = self._current_run.next_run
    self._current_run.next_run = new_run
    splitPoint = textPosition - self._current_run.text_start
    new_run.text_start += splitPoint
    new_run.text_length -= splitPoint
    self._current_run.text_length = splitPoint
    self._current_run = new_run