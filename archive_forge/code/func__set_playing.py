from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def _set_playing(self, playing: bool) -> None:
    starting = not self._playing and playing
    self._playing = playing
    source = self.source
    if playing and source:
        if source.audio_format is not None:
            if (was_created := (self._audio_player is None)):
                self._create_audio_player()
            if self._audio_player is not None and (was_created or starting):
                self._audio_player.prefill_audio()
        if bl.logger is not None:
            bl.logger.init_wall_time()
            bl.logger.log('p.P._sp', 0.0)
        if source.video_format is not None:
            if self._texture is None:
                self._create_texture()
        if self._audio_player is not None:
            self._audio_player.play()
        if source.video_format is not None:
            pyglet.clock.schedule_once(self.update_texture, 0)
        self._timer.start()
        if self._audio_player is None and source.video_format is None:
            pyglet.clock.schedule_once(lambda dt: self.dispatch_event('on_eos'), source.duration)
    else:
        if self._audio_player is not None:
            self._audio_player.stop()
        pyglet.clock.unschedule(self.update_texture)
        self._timer.pause()