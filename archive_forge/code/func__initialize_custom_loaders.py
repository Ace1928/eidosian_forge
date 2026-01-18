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
@classmethod
def _initialize_custom_loaders(cls):
    """Initialize the loaders needed to load custom fonts."""
    if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
        cls._font_loader = IDWriteInMemoryFontFileLoader()
        cls._write_factory.CreateInMemoryFontFileLoader(byref(cls._font_loader))
        cls._write_factory.RegisterFontFileLoader(cls._font_loader)
        cls._font_builder = IDWriteFontSetBuilder1()
        cls._write_factory.CreateFontSetBuilder1(byref(cls._font_builder))
    else:
        cls._font_loader = LegacyFontFileLoader()
        cls._write_factory.RegisterFontFileLoader(cls._font_loader.as_interface(IDWriteFontFileLoader_LI))
        cls._font_collection_loader = LegacyCollectionLoader(cls._write_factory, cls._font_loader)
        cls._write_factory.RegisterFontCollectionLoader(cls._font_collection_loader)
        cls._font_loader_key = cast(create_unicode_buffer('legacy_font_loader'), c_void_p)