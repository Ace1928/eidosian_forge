from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def _create_texture(self) -> None:
    video_format = self.source.video_format
    self._texture = pyglet.image.Texture.create(video_format.width, video_format.height, GL_TEXTURE_2D)
    self._texture = self._texture.get_transform(flip_y=True)
    self._texture.anchor_y = 0
    return self._texture