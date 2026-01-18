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
def _get_image(self):
    return self._image