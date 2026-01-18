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
class WMFSource(Source):
    low_latency = True
    decode_audio = True
    decode_video = True

    def __init__(self, filename, file=None):
        assert any([self.decode_audio, self.decode_video]), 'Source must decode audio, video, or both, not none.'
        self._current_video_sample = None
        self._current_video_buffer = None
        self._timestamp = 0
        self._attributes = None
        self._stream_obj = None
        self._imf_bytestream = None
        self._wfx = None
        self._stride = None
        self.set_config_attributes()
        self._source_reader = IMFSourceReader()
        if file is not None:
            data = file.read()
            self._imf_bytestream = IMFByteStream()
            data_len = len(data)
            if WINDOWS_7_OR_GREATER:
                hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, data_len)
                ptr = kernel32.GlobalLock(hglob)
                ctypes.memmove(ptr, data, data_len)
                kernel32.GlobalUnlock(hglob)
                self._stream_obj = com.pIUnknown()
                ole32.CreateStreamOnHGlobal(hglob, True, ctypes.byref(self._stream_obj))
                MFCreateMFByteStreamOnStream(self._stream_obj, ctypes.byref(self._imf_bytestream))
            else:
                MFCreateTempFile(MF_ACCESSMODE_READWRITE, MF_OPENMODE_DELETE_IF_EXIST, MF_FILEFLAGS_NONE, ctypes.byref(self._imf_bytestream))
                wrote_length = ULONG()
                data_ptr = cast(data, POINTER(BYTE))
                self._imf_bytestream.Write(data_ptr, data_len, ctypes.byref(wrote_length))
                self._imf_bytestream.SetCurrentPosition(0)
                if wrote_length.value != data_len:
                    raise DecodeException('Could not write all of the data to the bytestream file.')
            try:
                MFCreateSourceReaderFromByteStream(self._imf_bytestream, self._attributes, ctypes.byref(self._source_reader))
            except OSError as err:
                raise DecodeException(err) from None
        else:
            try:
                MFCreateSourceReaderFromURL(filename, self._attributes, ctypes.byref(self._source_reader))
            except OSError as err:
                raise DecodeException(err) from None
        if self.decode_audio:
            self._load_audio()
        if self.decode_video:
            self._load_video()
        assert self.audio_format or self.video_format, 'Source was decoded, but no video or audio streams were found.'
        try:
            prop = PROPVARIANT()
            self._source_reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE, ctypes.byref(MF_PD_DURATION), ctypes.byref(prop))
            self._duration = timestamp_from_wmf(prop.llVal)
            ole32.PropVariantClear(ctypes.byref(prop))
        except OSError:
            warnings.warn("Could not determine duration of media file: '{}'.".format(filename))

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

    def get_format(self):
        """Returns the WAVEFORMATEX data which has more information thah audio_format"""
        return self._wfx

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

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        flags = DWORD()
        timestamp = ctypes.c_longlong()
        imf_sample = IMFSample()
        imf_buffer = IMFMediaBuffer()
        while True:
            self._source_reader.ReadSample(self._audio_stream_index, 0, None, ctypes.byref(flags), ctypes.byref(timestamp), ctypes.byref(imf_sample))
            if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
                assert _debug('WMFAudioDecoder: Data is no longer valid.')
                break
            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                assert _debug('WMFAudioDecoder: End of data from stream source.')
                break
            if not imf_sample:
                assert _debug('WMFAudioDecoder: No sample.')
                continue
            imf_sample.ConvertToContiguousBuffer(ctypes.byref(imf_buffer))
            audio_data_ptr = POINTER(BYTE)()
            audio_data_length = DWORD()
            imf_buffer.Lock(ctypes.byref(audio_data_ptr), None, ctypes.byref(audio_data_length))
            audio_data = ctypes.string_at(audio_data_ptr, audio_data_length.value)
            imf_buffer.Unlock()
            imf_buffer.Release()
            imf_sample.Release()
            return AudioData(audio_data, audio_data_length.value, timestamp_from_wmf(timestamp.value), audio_data_length.value / self.audio_format.sample_rate, [])
        return None

    def get_next_video_frame(self, skip_empty_frame=True):
        video_data_length = DWORD()
        flags = DWORD()
        timestamp = ctypes.c_longlong()
        if self._current_video_sample:
            self._current_video_buffer.Release()
            self._current_video_sample.Release()
        self._current_video_sample = IMFSample()
        self._current_video_buffer = IMFMediaBuffer()
        while True:
            self._source_reader.ReadSample(self._video_stream_index, 0, None, ctypes.byref(flags), ctypes.byref(timestamp), ctypes.byref(self._current_video_sample))
            if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
                assert _debug('WMFVideoDecoder: Data is no longer valid.')
                new = IMFMediaType()
                self._source_reader.GetCurrentMediaType(self._video_stream_index, ctypes.byref(new))
                stride = ctypes.c_uint32()
                new.GetUINT32(MF_MT_DEFAULT_STRIDE, ctypes.byref(stride))
                new.Release()
                self._stride = stride.value
            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                self._timestamp = None
                assert _debug('WMFVideoDecoder: End of data from stream source.')
                break
            if not self._current_video_sample:
                assert _debug('WMFVideoDecoder: No sample.')
                continue
            self._current_video_buffer = IMFMediaBuffer()
            self._current_video_sample.ConvertToContiguousBuffer(ctypes.byref(self._current_video_buffer))
            video_data = POINTER(BYTE)()
            self._current_video_buffer.Lock(ctypes.byref(video_data), None, ctypes.byref(video_data_length))
            width = self.video_format.width
            height = self.video_format.height
            self._timestamp = timestamp_from_wmf(timestamp.value)
            self._current_video_buffer.Unlock()
            return image.ImageData(width, height, 'BGRA', video_data, self._stride)
        return None

    def get_next_video_timestamp(self):
        return self._timestamp

    def seek(self, timestamp):
        timestamp = min(timestamp, self._duration) if self._duration else timestamp
        prop = PROPVARIANT()
        prop.vt = VT_I8
        prop.llVal = timestamp_to_wmf(timestamp)
        pos_com = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        try:
            self._source_reader.SetCurrentPosition(pos_com, prop)
        except OSError as err:
            warnings.warn(str(err))
        ole32.PropVariantClear(ctypes.byref(prop))

    @staticmethod
    def _get_attribute_size(attributes, guidKey):
        """ Convert int64 attributes to int32"""
        size = ctypes.c_uint64()
        attributes.GetUINT64(guidKey, size)
        lParam = size.value
        x = ctypes.c_int32(lParam).value
        y = ctypes.c_int32(lParam >> 32).value
        return (x, y)

    def set_config_attributes(self):
        """ Here we set user specified attributes, by default we try to set low latency mode. (Win7+)"""
        if self.low_latency or self.decode_video:
            self._attributes = IMFAttributes()
            MFCreateAttributes(ctypes.byref(self._attributes), 3)
        if self.low_latency and WINDOWS_7_OR_GREATER:
            self._attributes.SetUINT32(ctypes.byref(MF_LOW_LATENCY), 1)
            assert _debug('WMFAudioDecoder: Setting configuration attributes.')
        if self.decode_video:
            self._attributes.SetUINT32(ctypes.byref(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS), 1)
            self._attributes.SetUINT32(ctypes.byref(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING), 1)
            assert _debug('WMFVideoDecoder: Setting configuration attributes.')

    def __del__(self):
        if self._source_reader:
            self._source_reader.Release()
        if self._stream_obj:
            self._stream_obj.Release()
        if self._imf_bytestream:
            self._imf_bytestream.Release()
        if self._current_video_sample:
            self._current_video_buffer.Release()
            self._current_video_sample.Release()