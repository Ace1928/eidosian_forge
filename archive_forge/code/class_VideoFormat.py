import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class VideoFormat:
    """Video details.

    An instance of this class is provided by sources with a video stream. You
    should not modify the fields.

    Note that the sample aspect has no relation to the aspect ratio of the
    video image.  For example, a video image of 640x480 with sample aspect 2.0
    should be displayed at 1280x480.  It is the responsibility of the
    application to perform this scaling.

    Args:
        width (int): Width of video image, in pixels.
        height (int): Height of video image, in pixels.
        sample_aspect (float): Aspect ratio (width over height) of a single
            video pixel.
        frame_rate (float): Frame rate (frames per second) of the video.

            .. versionadded:: 1.2
    """

    def __init__(self, width: int, height: int, sample_aspect: float=1.0) -> None:
        self.width = width
        self.height = height
        self.sample_aspect = sample_aspect
        self.frame_rate = None

    def __eq__(self, other) -> bool:
        if isinstance(other, VideoFormat):
            return self.width == other.width and self.height == other.height and (self.sample_aspect == other.sample_aspect) and (self.frame_rate == other.frame_rate)
        return False