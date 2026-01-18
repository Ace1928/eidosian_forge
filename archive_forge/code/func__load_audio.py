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
def _load_audio(self, stream=MF_SOURCE_READER_FIRST_AUDIO_STREAM):
    """ Prepares the audio stream for playback by detecting if it's compressed and attempting to decompress to PCM.
            Default: Only get the first available audio stream.
        """
    self._audio_stream_index = stream
    imfmedia = IMFMediaType()
    try:
        self._source_reader.GetNativeMediaType(self._audio_stream_index, 0, ctypes.byref(imfmedia))
    except OSError as err:
        if err.winerror == MF_E_INVALIDSTREAMNUMBER:
            assert _debug('WMFAudioDecoder: No audio stream found.')
        return
    guid_audio_type = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    imfmedia.GetGUID(MF_MT_MAJOR_TYPE, ctypes.byref(guid_audio_type))
    if guid_audio_type == MFMediaType_Audio:
        assert _debug('WMFAudioDecoder: Found Audio Stream.')
        if not self.decode_video:
            self._source_reader.SetStreamSelection(MF_SOURCE_READER_ANY_STREAM, False)
        self._source_reader.SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM, True)
        guid_compressed = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        imfmedia.GetGUID(MF_MT_SUBTYPE, ctypes.byref(guid_compressed))
        if guid_compressed == MFAudioFormat_PCM or guid_compressed == MFAudioFormat_Float:
            assert _debug(f'WMFAudioDecoder: Found Uncompressed Audio: {guid_compressed}')
        else:
            assert _debug(f'WMFAudioDecoder: Found Compressed Audio: {guid_compressed}')
            mf_mediatype = IMFMediaType()
            MFCreateMediaType(ctypes.byref(mf_mediatype))
            mf_mediatype.SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio)
            mf_mediatype.SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM)
            try:
                self._source_reader.SetCurrentMediaType(self._audio_stream_index, None, mf_mediatype)
            except OSError as err:
                raise DecodeException(err) from None
        decoded_media_type = IMFMediaType()
        self._source_reader.GetCurrentMediaType(self._audio_stream_index, ctypes.byref(decoded_media_type))
        wfx_length = ctypes.c_uint32()
        wfx = POINTER(WAVEFORMATEX)()
        MFCreateWaveFormatExFromMFMediaType(decoded_media_type, ctypes.byref(wfx), ctypes.byref(wfx_length), 0)
        self._wfx = wfx.contents
        self.audio_format = AudioFormat(channels=self._wfx.nChannels, sample_size=self._wfx.wBitsPerSample, sample_rate=self._wfx.nSamplesPerSec)
    else:
        assert _debug('WMFAudioDecoder: Audio stream not found')