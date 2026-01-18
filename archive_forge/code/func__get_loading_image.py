from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.image import ImageLoader, Image
from kivy.config import Config
from kivy.utils import platform
from collections import deque
from time import sleep
from os.path import join
from os import write, close, unlink, environ
import threading
import mimetypes
def _get_loading_image(self):
    if not self._loading_image:
        loading_png_fn = join(kivy_data_dir, 'images', 'image-loading.zip')
        self._loading_image = ImageLoader.load(filename=loading_png_fn)
    return self._loading_image