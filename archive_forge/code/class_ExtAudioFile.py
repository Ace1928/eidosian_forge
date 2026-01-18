import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
class ExtAudioFile(AudioFile):
    """A CoreAudio "extended audio file". Reads information and raw PCM
    audio data from any file that CoreAudio knows how to decode.

        >>> with ExtAudioFile('something.m4a') as f:
        >>>     print f.samplerate
        >>>     print f.channels
        >>>     print f.duration
        >>>     for block in f:
        >>>         do_something(block)

    """

    def __init__(self, filename):
        url = CFURL(filename)
        try:
            self._obj = self._open_url(url)
        except:
            self.closed = True
            raise
        del url
        self.closed = False
        self._file_fmt = None
        self._client_fmt = None
        self.setup()

    @classmethod
    def _open_url(cls, url):
        """Given a CFURL Python object, return an opened ExtAudioFileRef.
        """
        file_obj = ctypes.c_void_p()
        check(_coreaudio.ExtAudioFileOpenURL(url._obj, ctypes.byref(file_obj)))
        return file_obj

    def set_client_format(self, desc):
        """Get the client format description. This describes the
        encoding of the data that the program will read from this
        object.
        """
        assert desc.mFormatID == AUDIO_ID_PCM
        check(_coreaudio.ExtAudioFileSetProperty(self._obj, PROP_CLIENT_DATA_FORMAT, ctypes.sizeof(desc), ctypes.byref(desc)))
        self._client_fmt = desc

    def get_file_format(self):
        """Get the file format description. This describes the type of
        data stored on disk.
        """
        if self._file_fmt is not None:
            return self._file_fmt
        desc = AudioStreamBasicDescription()
        size = ctypes.c_int(ctypes.sizeof(desc))
        check(_coreaudio.ExtAudioFileGetProperty(self._obj, PROP_FILE_DATA_FORMAT, ctypes.byref(size), ctypes.byref(desc)))
        self._file_fmt = desc
        return desc

    @property
    def channels(self):
        """The number of channels in the audio source."""
        return int(self.get_file_format().mChannelsPerFrame)

    @property
    def samplerate(self):
        """Gets the sample rate of the audio."""
        return int(self.get_file_format().mSampleRate)

    @property
    def duration(self):
        """Gets the length of the file in seconds (a float)."""
        return float(self.nframes) / self.samplerate

    @property
    def nframes(self):
        """Gets the number of frames in the source file."""
        length = ctypes.c_long()
        size = ctypes.c_int(ctypes.sizeof(length))
        check(_coreaudio.ExtAudioFileGetProperty(self._obj, PROP_LENGTH, ctypes.byref(size), ctypes.byref(length)))
        return length.value

    def setup(self, bitdepth=16):
        """Set the client format parameters, specifying the desired PCM
        audio data format to be read from the file. Must be called
        before reading from the file.
        """
        fmt = self.get_file_format()
        newfmt = copy.copy(fmt)
        newfmt.mFormatID = AUDIO_ID_PCM
        newfmt.mFormatFlags = PCM_IS_SIGNED_INT | PCM_IS_PACKED
        newfmt.mBitsPerChannel = bitdepth
        newfmt.mBytesPerPacket = fmt.mChannelsPerFrame * newfmt.mBitsPerChannel // 8
        newfmt.mFramesPerPacket = 1
        newfmt.mBytesPerFrame = newfmt.mBytesPerPacket
        self.set_client_format(newfmt)

    def read_data(self, blocksize=4096):
        """Generates byte strings reflecting the audio data in the file.
        """
        frames = ctypes.c_uint(blocksize // self._client_fmt.mBytesPerFrame)
        buf = ctypes.create_string_buffer(blocksize)
        buflist = AudioBufferList()
        buflist.mNumberBuffers = 1
        buflist.mBuffers[0].mNumberChannels = self._client_fmt.mChannelsPerFrame
        buflist.mBuffers[0].mDataByteSize = blocksize
        buflist.mBuffers[0].mData = ctypes.cast(buf, ctypes.c_void_p)
        while True:
            check(_coreaudio.ExtAudioFileRead(self._obj, ctypes.byref(frames), ctypes.byref(buflist)))
            assert buflist.mNumberBuffers == 1
            size = buflist.mBuffers[0].mDataByteSize
            if not size:
                break
            data = ctypes.cast(buflist.mBuffers[0].mData, ctypes.POINTER(ctypes.c_char))
            blob = data[:size]
            yield blob

    def close(self):
        """Close the audio file and free associated memory."""
        if not self.closed:
            check(_coreaudio.ExtAudioFileDispose(self._obj))
            self.closed = True

    def __del__(self):
        if _coreaudio:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.read_data()