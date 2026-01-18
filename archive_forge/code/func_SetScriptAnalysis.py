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
def SetScriptAnalysis(self, textPosition, textLength, scriptAnalysis):
    self.SetCurrentRun(textPosition)
    self.SplitCurrentRun(textPosition)
    while textLength > 0:
        run, textLength = self.FetchNextRun(textLength)
        run.script.script = scriptAnalysis[0].script
        run.script.shapes = scriptAnalysis[0].shapes
        self._script = run.script
    return 0