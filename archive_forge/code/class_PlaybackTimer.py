from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
class PlaybackTimer:
    """Playback Timer.

    This is a simple timer object which tracks the time elapsed. It can be
    paused and reset.
    """

    def __init__(self) -> None:
        self._elapsed = 0.0
        self._started_at = None

    def start(self) -> None:
        """Start the timer."""
        if self._started_at is None:
            self._started_at = time.perf_counter()

    def pause(self) -> None:
        """Pause the timer."""
        self._elapsed = self.get_time()
        self._started_at = None

    def reset(self) -> None:
        """Reset the timer to 0."""
        self._elapsed = 0.0
        if self._started_at is not None:
            self._started_at = time.perf_counter()

    def get_time(self) -> float:
        """Get the elapsed time."""
        if self._started_at is None:
            return self._elapsed
        return time.perf_counter() - self._started_at + self._elapsed

    def set_time(self, value: float) -> None:
        """
        Manually set the elapsed time.

        Args:
            value (float): the new elapsed time value
        """
        self.reset()
        self._elapsed = value