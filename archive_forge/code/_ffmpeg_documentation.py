from __future__ import unicode_literals
from past.builtins import basestring
from ._utils import basestring
from .nodes import (
Output file URL

    Syntax:
        `ffmpeg.output(stream1[, stream2, stream3...], filename, **ffmpeg_args)`

    Any supplied keyword arguments are passed to ffmpeg verbatim (e.g.
    ``t=20``, ``f='mp4'``, ``acodec='pcm'``, ``vcodec='rawvideo'``,
    etc.).  Some keyword-arguments are handled specially, as shown below.

    Args:
        video_bitrate: parameter for ``-b:v``, e.g. ``video_bitrate=1000``.
        audio_bitrate: parameter for ``-b:a``, e.g. ``audio_bitrate=200``.
        format: alias for ``-f`` parameter, e.g. ``format='mp4'``
            (equivalent to ``f='mp4'``).

    If multiple streams are provided, they are mapped to the same
    output.

    To tell ffmpeg to write to stdout, use ``pipe:`` as the filename.

    Official documentation: `Synopsis <https://ffmpeg.org/ffmpeg.html#Synopsis>`__
    