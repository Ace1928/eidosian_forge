from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class FrameParameters(object):
    """Information about a zstd frame.

    Instances have the following attributes:

    ``content_size``
       Integer size of original, uncompressed content. This will be ``0`` if the
       original content size isn't written to the frame (controlled with the
       ``write_content_size`` argument to ``ZstdCompressor``) or if the input
       content size was ``0``.

    ``window_size``
       Integer size of maximum back-reference distance in compressed data.

    ``dict_id``
       Integer of dictionary ID used for compression. ``0`` if no dictionary
       ID was used or if the dictionary ID was ``0``.

    ``has_checksum``
       Bool indicating whether a 4 byte content checksum is stored at the end
       of the frame.
    """

    def __init__(self, fparams):
        self.content_size = fparams.frameContentSize
        self.window_size = fparams.windowSize
        self.dict_id = fparams.dictID
        self.has_checksum = bool(fparams.checksumFlag)