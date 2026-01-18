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
class MyEnumerator(com.COMObject):
    _interfaces_ = [IDWriteFontFileEnumerator]

    def __init__(self, factory, loader):
        super().__init__()
        self.factory = cast(factory, IDWriteFactory)
        self.key = 'pyglet_dwrite'
        self.size = len(self.key)
        self.current_index = -1
        self._keys = []
        self._font_data = []
        self._font_files = []
        self._current_file = None
        self._font_key_ref = create_unicode_buffer('none')
        self._font_key_len = len(self._font_key_ref)
        self._file_loader = loader

    def AddFontData(self, fonts):
        self._font_data = fonts

    def MoveNext(self, hasCurrentFile):
        self.current_index += 1
        if self.current_index != len(self._font_data):
            font_file = IDWriteFontFile()
            self._file_loader.SetCurrentFont(self.current_index, self._font_data[self.current_index])
            key = self.current_index
            if not self.current_index in self._keys:
                buffer = pointer(c_uint32(key))
                ptr = cast(buffer, c_void_p)
                self._keys.append(ptr)
            self.factory.CreateCustomFontFileReference(self._keys[self.current_index], sizeof(buffer), self._file_loader, byref(font_file))
            self._font_files.append(font_file)
            hasCurrentFile[0] = 1
        else:
            hasCurrentFile[0] = 0
        pass

    def GetCurrentFontFile(self, fontFile):
        fontFile = cast(fontFile, POINTER(IDWriteFontFile))
        fontFile[0] = self._font_files[self.current_index]
        return 0