import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def _load_video(self, stream=MF_SOURCE_READER_FIRST_VIDEO_STREAM):
    self._video_stream_index = stream
    imfmedia = IMFMediaType()
    try:
        self._source_reader.GetCurrentMediaType(self._video_stream_index, ctypes.byref(imfmedia))
    except OSError as err:
        if err.winerror == MF_E_INVALIDSTREAMNUMBER:
            assert _debug('WMFVideoDecoder: No video stream found.')
        return
    assert _debug('WMFVideoDecoder: Found Video Stream')
    uncompressed_mt = IMFMediaType()
    MFCreateMediaType(ctypes.byref(uncompressed_mt))
    imfmedia.CopyAllItems(uncompressed_mt)
    imfmedia.Release()
    uncompressed_mt.SetGUID(MF_MT_SUBTYPE, MFVideoFormat_ARGB32)
    uncompressed_mt.SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)
    uncompressed_mt.SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, 1)
    try:
        self._source_reader.SetCurrentMediaType(self._video_stream_index, None, uncompressed_mt)
    except OSError as err:
        raise DecodeException(err) from None
    height, width = self._get_attribute_size(uncompressed_mt, MF_MT_FRAME_SIZE)
    self.video_format = VideoFormat(width=width, height=height)
    assert _debug('WMFVideoDecoder: Frame width: {} height: {}'.format(width, height))
    den, num = self._get_attribute_size(uncompressed_mt, MF_MT_FRAME_RATE)
    self.video_format.frame_rate = num / den
    assert _debug('WMFVideoDecoder: Frame Rate: {} / {} = {}'.format(num, den, self.video_format.frame_rate))
    if self.video_format.frame_rate < 0:
        self.video_format.frame_rate = 30000 / 1001
        assert _debug('WARNING: Negative frame rate, attempting to use default, but may experience issues.')
    den, num = self._get_attribute_size(uncompressed_mt, MF_MT_PIXEL_ASPECT_RATIO)
    self.video_format.sample_aspect = num / den
    assert _debug('WMFVideoDecoder: Pixel Ratio: {} / {} = {}'.format(num, den, self.video_format.sample_aspect))