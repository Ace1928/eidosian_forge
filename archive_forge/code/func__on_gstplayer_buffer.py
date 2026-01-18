from kivy.graphics.texture import Texture
from kivy.core.video import VideoBase
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.compat import PY2
from threading import Lock
from functools import partial
from os.path import realpath
from weakref import ref
def _on_gstplayer_buffer(video, width, height, data):
    video = video()
    if not video:
        return
    with video._buffer_lock:
        video._buffer = (width, height, data)