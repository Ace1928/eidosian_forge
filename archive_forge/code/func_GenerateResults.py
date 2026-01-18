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
def GenerateResults(self, analyzer, text, text_length):
    self._text = text
    self._textstart = 0
    self._textlength = text_length
    self._glyphstart = 0
    self._glyphcount = 0
    self._ptrs.clear()
    self._start_run = Run()
    self._start_run.text_length = text_length
    self._current_run = self._start_run
    analyzer.AnalyzeScript(self, 0, text_length, self)