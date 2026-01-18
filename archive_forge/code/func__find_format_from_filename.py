import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
def _find_format_from_filename(self, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext in {'bmp', 'jpe', 'lbm', 'pcx', 'png', 'pnm', 'tga', 'tiff', 'webp', 'xcf', 'xpm', 'xv'}:
        return ext
    elif ext in ('jpg', 'jpeg'):
        return 'jpg'
    elif ext in ('b64', 'base64'):
        return 'base64'
    return None